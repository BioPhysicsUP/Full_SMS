import copy
import time
from typing import Union, List

from PyQt5.QtCore import QRunnable, pyqtSlot

import processes as prcs
from my_logger import setup_logger
from signals import WorkerSignals, ProcessThreadSignals
from smsh5 import H5dataset, Particle
from thread_commands import StatusCmd, ProgressCmd

logger = setup_logger(__name__)


class ProcessThread(QRunnable):
    """
    Worker thread
    """

    def __init__(self, num_processes: int = None,
                 tasks: Union[prcs.ProcessTask, List[prcs.ProcessTask]] = None,
                 signals: WorkerSignals = None,
                 task_buffer_size: int = None):
        super().__init__()
        self._processes = []
        self.task_queue = prcs.create_queue()
        self.result_queue = prcs.create_queue()
        self.progress_queue = prcs.create_queue()
        self.force_stop = False
        self.is_running = False

        if num_processes:
            # assert type(num_processes) is int, 'Provided num_processes is ' \
            #                                    'not int'
            if type(num_processes) is not int:
                raise TypeError("Provided num_processes must be of type int")
            self.num_processes = num_processes
        else:
            self.num_processes = prcs.get_max_num_processes()

        if not task_buffer_size:
            task_buffer_size = self.num_processes
        self.task_buffer_size = task_buffer_size

        if not signals:
            self.signals = ProcessThreadSignals()
        else:
            # assert type(signals) is ThreadSignals, 'Provided signals wrong ' \
            #                                        'type'
            if type(signals) is not WorkerSignals:
                raise TypeError("Provided signals must be of type "
                                "ThreadSignals")
            self.signals = signals

        self.tasks = []
        if tasks:
            self.add_task(tasks)
        self.results = []

    def add_tasks(self, tasks: Union[prcs.ProcessTask,
                                     List[prcs.ProcessTask]]):
        if type(tasks) is not List:
            tasks = [tasks]
        all_valid = all([type(task) is prcs.ProcessTask for task in tasks])
        # assert all_valid, "At least some provided tasks are not correct type"
        if not all_valid:
            raise TypeError("At least some of provided tasks are not of "
                            "type ProcessTask")
        self.tasks.extend(tasks)

    def add_tasks_from_methods(self, objects: Union[object, List[object]],
                               method_name: str):
        if type(objects) is not list:
            objects = [objects]
        # assert type(method_name) is str, 'Method_name is not str'
        if type(method_name) is not str:
            raise TypeError("Provided method_name must be of type str")

        all_valid = all([hasattr(obj, method_name) for obj in objects])
        # assert all_valid, 'Some or all objects do not have specified method'
        if not all_valid:
            raise TypeError("Some or all objects do not have " "the specified method")

        for obj in objects:
            self.tasks.append(prcs.ProcessTask(obj=obj,
                                               method_name=method_name))

    @pyqtSlot()
    def run(self, num_processes: int = None):
        """
        Your code goes in this function
        """

        self.is_running = True
        num_active_processes = 0
        try:
            self.results = [None]*len(self.tasks)
            self.signals.status_update.emit(StatusCmd.ShowMessage,
                                            'Busy with generic worker')
            prog_tracker = prcs.ProgressTracker(len(self.tasks))
            self.signals.progress.emit(ProgressCmd.SetMax, 100)
            num_init_tasks = len(self.tasks)
            task_uuids = [task.uuid for task in self.tasks]
            # assert num_init_tasks, 'No tasks were provided'
            if not num_init_tasks: raise TypeError("No tasks were provided")

            num_used_processes = self.num_processes
            if num_init_tasks < self.num_processes:
                num_used_processes = num_init_tasks
            num_active_processes = 0
            for _ in range(num_used_processes):
                process = prcs.SingleProcess(task_queue=self.task_queue,
                                             result_queue=self.result_queue)
                self._processes.append(process)
                process.start()
                num_active_processes += 1

            num_task_left = len(self.tasks)
            tasks_todo = copy.copy(self.tasks)

            init_num = num_used_processes + self.task_buffer_size
            rest = len(tasks_todo) - init_num
            if rest < 0:
                init_num += rest

            for _ in range(init_num):
                self.task_queue.put(tasks_todo.pop(0))

            while num_task_left and not self.force_stop:
                try:
                    result = self.result_queue.get(timeout=1)
                except prcs.get_empty_queue_exception():
                    pass
                else:
                    if len(tasks_todo):
                        self.task_queue.put(tasks_todo.pop(0))

                    if type(result) is not prcs.ProcessTaskResult:
                        raise TypeError("Task result is not of type "
                                        "ProcessTaskResult")

                    ind = task_uuids.index(result.task_uuid)
                    # self.tasks[ind].obj = result.new_task_obj
                    self.results[ind] = result
                    self.result_queue.task_done()
                    num_task_left -= 1
                    prog_value = prog_tracker.iterate()
                    if prog_value:
                        self.signals.progress.emit(ProgressCmd.Step,
                                                   prog_value)

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
            if len(self._processes):
                for _ in range(num_used_processes):
                    self.task_queue.put(None)
                for _ in range(num_used_processes):
                    if self.result_queue.get() is True:
                        self.result_queue.task_done()
                        num_active_processes -= 1
                self.task_queue.close()
                self.result_queue.close()
                while any([p.is_alive() for p in self._processes]):
                    time.sleep(1)

            self.signals.result.emit(self.results)
            self.signals.finished.emit(self)
            self.signals.status_update.emit(StatusCmd.Reset, '')
            self.signals.progress.emit(ProgressCmd.Complete, 0)
            self.is_running = False


class WorkerFitLifetimes(QRunnable):
    """ A QRunnable class to create a worker thread for fitting lifetimes. """

    def __init__(self, fit_lifetimes_func, data, currentparticle, fitparam, mode: str,
                 resolve_selected=None) -> None:
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
        """ The code that will be run when the thread is started. """

        try:
            self.fit_lifetimes_func(self.signals.start_progress, self.signals.progress,
                                    self.signals.status_message, self.signals.reset_gui,
                                    self.data, self.currentparticle, self.fitparam,
                                    self.mode, self.resolve_selected)
        except Exception as err:
            self.signals.error.emit(err)
        finally:
            self.signals.fitting_finished.emit(self.mode)


class WorkerGrouping(QRunnable):

    def __init__(self,
                 data: H5dataset,
                 grouping_func,
                 mode: str,
                 currentparticle: Particle = None,
                 group_selected=None) -> None:
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
        """ The code that will be run when the thread is started. """

        try:
            self.grouping_func(start_progress_sig=self.signals.start_progress,
                               progress_sig=self.signals.progress,
                               status_sig=self.signals.status_message,
                               reset_gui_sig=self.signals.reset_gui,
                               data=self.data,
                               mode=self.mode,
                               currentparticle=self.currentparticle,
                               group_selected=self.group_selected)
        except Exception as err:
            self.signals.error.emit(err)
        finally:
            self.signals.grouping_finished.emit(self.mode)
            pass


class WorkerResolveLevels(QRunnable):
    """ A QRunnable class to create a worker thread for resolving levels. """

    def __init__(self, resolve_levels_func, conf: Union[int, float], data: H5dataset,
                 currentparticle: Particle,
                 mode: str,
                 resolve_selected=None,
                 end_time_s=None) -> None:
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
        """ The code that will be run when the thread is started. """

        try:
            self.resolve_levels_func(self.signals.start_progress, self.signals.progress,
                                     self.signals.status_message, self.signals.reset_gui,
                                     self.signals.level_resolved,
                                     self.conf, self.data, self.currentparticle,
                                     self.mode, self.resolve_selected, self.end_time_s)
        except Exception as err:
            self.signals.error.emit(err)
        finally:
            self.signals.resolve_finished.emit(self.mode)


class WorkerBinAll(QRunnable):
    """ A QRunnable class to create a worker thread for binning all the data. """

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
        """ The code that will be run when the thread is started. """

        try:
            self.binall_func(self.dataset, self.bin_size, self.signals.start_progress,
                             self.signals.progress, self.signals.status_message,
                             self.signals.bin_size)
        except Exception as err:
            self.signals.error.emit(err)
        finally:
            # self.signals.resolve_finished.emit(False)  ?????
            pass