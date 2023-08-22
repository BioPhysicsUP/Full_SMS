# import copy
from __future__ import annotations
import time
from typing import Union, List, TYPE_CHECKING

from PyQt5.QtCore import QRunnable, pyqtSlot

# import processes as prcs
from my_logger import setup_logger
from signals import WorkerSignals, ProcessThreadSignals, worker_sig_pass
from smsh5 import H5dataset, Particle

# from thread_commands import StatusCmd, ProgressCmd
from processes import (
    ProcessProgressCmd as PPCmd,
    ProcessSigPassTask as PSCmd,
    ProcessProgressTask as PPTask,
    ProcessSigPassTask as PSTask,
    ProcessTask,
    create_queue,
    get_max_num_processes,
    get_empty_queue_exception,
    SingleProcess,
    ProcessTaskResult,
    prog_sig_pass,
    ProcessProgress,
    ProcessProgFeedback,
    apply_autoproxy_fix,
    create_manager,
)
from tempfile import TemporaryDirectory

if TYPE_CHECKING:
    from generate_sums import CPSums

logger = setup_logger(__name__)


class ProcessThread(QRunnable):
    """
    Worker thread
    """

    def __init__(
        self,
        num_processes: int = None,
        tasks: Union[ProcessTask, List[ProcessTask]] = None,
        signals: ProcessThreadSignals = None,
        worker_signals: WorkerSignals = None,
        task_buffer_size: int = None,
        status_message: str = None,
        temp_dir: TemporaryDirectory = None,
    ):
        # logger.info("Inside ProcessThread __init__")
        super().__init__()
        # logger.info("After super().__init__()")
        self._processes = []
        # logger.info("About to create manager")
        self._manager = create_manager()
        # logger.info("About to create queues")
        self.task_queue = create_queue()
        self.result_queue = create_queue()
        self.feedback_queue = self._manager.Queue()
        # self.feedback_queue = create_queue()

        self.force_stop = False
        self.is_running = False
        self._status_message = status_message
        self._temp_dir = temp_dir

        if num_processes:
            # assert type(num_processes) is int, 'Provided num_processes is ' \
            #                                    'not int'
            if type(num_processes) is not int:
                raise TypeError("Provided num_processes must be of type int")
            self.num_processes = num_processes
        else:
            self.num_processes = get_max_num_processes() - 1

        if not task_buffer_size:
            task_buffer_size = self.num_processes // 2
        self.task_buffer_size = task_buffer_size

        if not signals:
            # logger.info("About to create ProcessThreadsSignals object")
            self.signals = ProcessThreadSignals()
        else:
            # assert type(signals) is ThreadSignals, 'Provided signals wrong ' \
            #                                        'type'
            if type(signals) is not ProcessThreadSignals:
                raise TypeError("Provided signals must be of type ProcessThreadSignals")
            self.signals = signals

        if not worker_signals:
            self.worker_signals = WorkerSignals()
        else:
            if type(worker_signals) is not WorkerSignals:
                raise TypeError("Provided signals must be of type ProcessThreadSignals")
            self.worker_signals = worker_signals

        # self.tasks = self._manager.list()
        self.tasks = list()
        if tasks:
            self.add_task(tasks)
        self.results = []

    @property
    def status_message(self):
        return self._status_message

    @status_message.setter
    def status_message(self, message: str):
        assert type(message) is str, "status_message must be str"
        self._status_message = message

    def add_tasks(self, tasks: Union[ProcessTask, List[ProcessTask]]):
        if type(tasks) is not List:
            tasks = [tasks]
        all_valid = all([type(task) is ProcessTask for task in tasks])
        # assert all_valid, "At least some provided tasks are not correct type"
        if not all_valid:
            raise TypeError("At least some of provided tasks are not of " "type ProcessTask")
        self.tasks.extend(tasks)

    def add_tasks_from_methods(self, objects: Union[object, List[object]], method_name: str, args=None):
        if type(objects) is not list:
            objects = [objects]
        # assert type(method_name) is str, 'Method_name is not str'
        if type(method_name) is not str:
            raise TypeError("Provided method_name must be of type str")

        all_valid = all([hasattr(obj, method_name) for obj in objects])
        # assert all_valid, 'Some or all objects do not have specified method'
        if not all_valid:
            raise TypeError("Some or all objects do not have " "the specified method")

        if args is not None and type(args) is not tuple:
            args = (args,)
        for obj in objects:
            self.tasks.append(ProcessTask(obj=obj, method_name=method_name, args=args))

    @pyqtSlot()
    def run(self, num_processes: int = None):
        """
        Your code goes in this function
        """

        logger.info("Running Process Thread")
        self.is_running = True
        num_active_processes = 0
        try:
            self.results = [None] * len(self.tasks)
            # self.signals.status_update.emit("Testing")
            # prog_tracker = prcs.ProgressTracker(len(self.tasks))

            num_init_tasks = len(self.tasks)
            if not num_init_tasks:
                raise TypeError("No tasks were provided")

            if num_init_tasks > 1:
                if self._status_message is None:
                    status_message = "Busy..."
                else:
                    status_message = self._status_message
                self.signals.status_update.emit(status_message)
                self.signals.start_progress.emit(num_init_tasks)
            elif self._status_message is not None:
                self.signals.status_update.emit(self._status_message)

            task_uuids = [task.uuid for task in self.tasks]
            num_used_processes = self.num_processes
            if num_init_tasks < self.num_processes:
                num_used_processes = num_init_tasks
            num_active_processes = 0
            for _ in range(num_used_processes):
                process = SingleProcess(
                    task_queue=self.task_queue,
                    result_queue=self.result_queue,
                    feedback_queue=self.feedback_queue,
                    temp_dir=self._temp_dir,
                )
                self._processes.append(process)
                process.start()
                num_active_processes += 1

            num_task_left = len(self.tasks)
            single_task = num_task_left == 1

            init_num = num_used_processes + self.task_buffer_size
            rest = len(self.tasks) - init_num
            if rest < 0:
                init_num += rest

            process_progress = None
            if not single_task:
                prog_fb = ProcessProgFeedback(feedback_queue=self.feedback_queue)
                process_progress = ProcessProgress(prog_fb=prog_fb, num_iterations=num_init_tasks)
                process_progress.start_progress()

            next_task_ind = 0
            for _ in range(init_num):
                next_task = self.tasks.pop(0)
                self.task_queue.put(next_task)
                del next_task
                next_task_ind += 1

            while num_task_left and not self.force_stop:
                try:
                    if not self.feedback_queue.empty():
                        while not self.feedback_queue.empty():
                            self.check_fbk_queue()

                    result = self.result_queue.get(timeout=0.01)
                except get_empty_queue_exception():
                    pass
                else:
                    # if next_task_ind != len(self.tasks):
                    if len(self.tasks):
                        self.task_queue.put(self.tasks.pop(0))
                        # self.task_queue.put(self.tasks[next_task_ind])
                        next_task_ind += 1

                    if type(result) is not ProcessTaskResult:
                        if isinstance(result, Exception):
                            raise result
                        else:
                            raise TypeError("Task result is not of type ProcessTaskResult")

                    elif not result.dont_send:
                        self.signals.results.emit(result)
                    del result

                    # ind = task_uuids.index(result.task_uuid)
                    # self.results[ind] = result

                    self.result_queue.task_done()
                    if not single_task:
                        process_progress.iterate()
                    num_task_left -= 1

        except Exception as exception:
            self.signals.error.emit(exception)
        # else:
        #     self.signals.result.emit(self.tasks)
        finally:
            if self.force_stop:
                while not self.task_queue.empty():
                    self.task_queue.get()
                    self.task_queue.task_done()
                while not self.result_queue.empty():
                    self.result_queue.get()
                    self.result_queue.task_done()
            else:
                time.sleep(1)
                try:
                    if not self.feedback_queue.empty():
                        for _ in range(self.feedback_queue.qsize()):
                            self.check_fbk_queue()
                except BrokenPipeError as exception:
                    print("Warning: Broken Pipe")
                    self.signals.error.emit(exception)
            if process in self._processes:
                for _ in range(num_used_processes):
                    self.task_queue.put(None)
                for _ in range(num_used_processes):
                    result = self.result_queue.get()
                    if result is True:
                        self.result_queue.task_done()
                        num_active_processes -= 1
                while any([p.is_alive() for p in self._processes]):
                    time.sleep(1)

            # self.signals.results.emit(self.results)
            self.signals.end_progress.emit()
            self.is_running = False
            self.signals.finished.emit(self)
            # self.task_queue.join()
            # self.result_queue.join()
            # self.feedback_queue.join()
            self._manager.shutdown()

    def check_fbk_queue(self):
        fbk_return = self.feedback_queue.get()
        if type(fbk_return) is PPTask:
            prog_sig_pass(signals=self.signals, cmd=fbk_return.task_cmd, args=fbk_return.args)
        elif type(fbk_return) is PSTask:
            worker_sig_pass(
                signals=self.worker_signals,
                sig_type=fbk_return.sig_pass_type,
                args=fbk_return.sig_args,
            )


class WorkerFitLifetimes(QRunnable):
    """A QRunnable class to create a worker thread for fitting lifetimes."""

    def __init__(
        self,
        fit_lifetimes_func,
        data,
        currentparticle,
        fitparam,
        mode: str,
        resolve_selected=None,
    ) -> None:
        """
        Initiate Resolve Levels Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to resolve a single,
        the selected, or all the particles'.

        Parameters
        ----------
        resolve_levels_func : function
            The function that will be called to perform the resolving of the levels.
        mode : {'current', 'selected', 'all'}
            Determines the mode that the levels need to be resolved on. Options are 'current', 'selected' or 'all'
        resolve_selected : list[smsh5.Particle], optional
            The provided instances of the class Particle in smsh5 will be resolved.
        """

        super(WorkerFitLifetimes, self).__init__()
        self.mode = mode
        self.signals = WorkerSignals()
        self.fit_lifetimes_func = fit_lifetimes_func
        self.resolve_selected = resolve_selected
        self.data = data
        self.currentparticle = currentparticle
        self.fitparam = fitparam

    @pyqtSlot()
    def run(self) -> None:
        """The code that will be run when the thread is started."""

        try:
            self.fit_lifetimes_func(
                self.signals.start_progress,
                self.signals.progress,
                self.signals.status_message,
                self.signals.reset_gui,
                self.data,
                self.currentparticle,
                self.fitparam,
                self.mode,
                self.resolve_selected,
            )
        except Exception as err:
            self.signals.error.emit(err)
        finally:
            self.signals.fitting_finished.emit(self.mode)


class WorkerGrouping(QRunnable):
    def __init__(
        self,
        data: H5dataset,
        grouping_func,
        mode: str,
        currentparticle: Particle = None,
        group_selected=None,
    ) -> None:
        """
        Initiate Resolve Levels Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to resolve a single,
        the selected, or all the particles'.

        Parameters
        ----------
        resolve_levels_func : function
            The function that will be called to perform the resolving of the levels.
        mode : {'current', 'selected', 'all'}
            Determines the mode that the levels need to be resolved on. Options are 'current', 'selected' or 'all'
        resolve_selected : list[smsh5.Particle], optional
            The provided instances of the class Particle in smsh5 will be resolved.
        """

        super(WorkerGrouping, self).__init__()
        self.mode = mode
        self.signals = WorkerSignals()
        self.grouping_func = grouping_func
        self.group_selected = group_selected
        self.data = data
        self.currentparticle = currentparticle
        # self.fitparam = fitparam

    @pyqtSlot()
    def run(self) -> None:
        """The code that will be run when the thread is started."""

        try:
            self.grouping_func(
                start_progress_sig=self.signals.start_progress,
                progress_sig=self.signals.progress,
                status_sig=self.signals.status_message,
                reset_gui_sig=self.signals.reset_gui,
                data=self.data,
                mode=self.mode,
                currentparticle=self.currentparticle,
                group_selected=self.group_selected,
            )
        except Exception as err:
            self.signals.error.emit(err)
        finally:
            self.signals.grouping_finished.emit(self.mode)
            pass


class WorkerResolveLevels(QRunnable):
    """A QRunnable class to create a worker thread for resolving levels."""

    def __init__(
        self,
        resolve_levels_func,
        conf: Union[int, float],
        data: H5dataset,
        currentparticle: Particle,
        mode: str,
        resolve_selected=None,
        end_time_s=None,
    ) -> None:
        """
        Initiate Resolve Levels Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to resolve a single,
        the selected, or all the particles'.

        Parameters
        ----------
        resolve_levels_func : function
            The function that will be called to perform the resolving of the levels.
        mode : {'current', 'selected', 'all'}
            Determines the mode that the levels need to be resolved on. Options are 'current', 'selected' or 'all'
        resolve_selected : list[smsh5.Particle], optional
            The provided instances of the class Particle in smsh5 will be resolved.
        """

        super(WorkerResolveLevels, self).__init__()
        self.mode = mode
        self.signals = WorkerSignals()
        self.resolve_levels_func = resolve_levels_func
        self.resolve_selected = resolve_selected
        self.conf = conf
        self.data = data
        self.currentparticle = currentparticle
        self.end_time_s = end_time_s
        # print(self.currentparticle)

    @pyqtSlot()
    def run(self) -> None:
        """The code that will be run when the thread is started."""

        try:
            self.resolve_levels_func(
                self.signals.start_progress,
                self.signals.progress,
                self.signals.status_message,
                self.signals.reset_gui,
                self.signals.level_resolved,
                self.conf,
                self.data,
                self.currentparticle,
                self.mode,
                self.resolve_selected,
                self.end_time_s,
            )
        except Exception as err:
            self.signals.error.emit(err)
        finally:
            self.signals.resolve_finished.emit(self.mode)


class WorkerBinAll(QRunnable):
    """A QRunnable class to create a worker thread for binning all the data."""

    def __init__(self, dataset, binall_func, bin_size):
        """
        Initiate Open File Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to open a h5 file
        and populate the tree in the mainwindow g

        Parameters
        ----------
        fname : str
            The name of the file.
        binall_func : function
            Function to be called that will read the h5 file and populate the tree on the g
        """

        super(WorkerBinAll, self).__init__()
        self.dataset = dataset
        self.binall_func = binall_func
        self.signals = WorkerSignals()
        self.bin_size = bin_size

    @pyqtSlot()
    def run(self) -> None:
        """The code that will be run when the thread is started."""

        try:
            self.binall_func(self.dataset, self.bin_size)
        except Exception as err:
            self.signals.error.emit(err)
        finally:
            # self.signals.resolve_finished.emit(False)  ?????
            pass
