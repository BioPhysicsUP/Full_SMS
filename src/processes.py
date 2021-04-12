from __future__ import  annotations
import multiprocessing as mp
from multiprocessing import managers
from enum import IntEnum, auto
from queue import Empty
from typing import List, Union, TYPE_CHECKING
from uuid import UUID, uuid1
import time
from numpy import ndarray

from my_logger import setup_logger
from signals import WorkerSigPassType, ProcessThreadSignals
from tree_model import DatasetTreeNode

from inspect import signature
from functools import wraps

from change_point import ChangePoints
from grouping import AHCA
# from smsh5 import Histogram

if TYPE_CHECKING:
    from smsh5 import H5dataset, Histogram

logger = setup_logger(__name__)
orig_AutoProxy = managers.AutoProxy


@wraps(managers.AutoProxy)
def AutoProxy(*args, incref=True, manager_owned=False, **kwargs):
    # Create the autoproxy without the manager_owned flag, then
    # update the flag on the generated instance. If the manager_owned flag
    # is set, `incref` is disabled, so set it to False here for the same
    # result.
    autoproxy_incref = False if manager_owned else incref
    proxy = orig_AutoProxy(*args, incref=autoproxy_incref, **kwargs)
    proxy._owned_by_manager = manager_owned
    return proxy


def apply_autoproxy_fix():
    if "manager_owned" in signature(managers.AutoProxy).parameters:
        return

    logger.debug("Patching multiprocessing.managers.AutoProxy to add manager_owned")
    managers.AutoProxy = AutoProxy

    # re-register any types already registered to SyncManager without a custom
    # proxy type, as otherwise these would all be using the old unpatched AutoProxy
    SyncManager = managers.SyncManager
    registry = managers.SyncManager._registry
    for typeid, (callable, exposed, method_to_typeid, proxytype) in registry.items():
        if proxytype is not orig_AutoProxy:
            continue
        create_method = hasattr(managers.SyncManager, typeid)
        SyncManager.register(
            typeid,
            callable=callable,
            exposed=exposed,
            method_to_typeid=method_to_typeid,
            create_method=create_method,
        )


apply_autoproxy_fix()


def create_manager() -> mp.Manager:
    manager = mp.Manager()
    apply_autoproxy_fix()
    return manager


def create_queue() -> mp.JoinableQueue:
    return mp.JoinableQueue()
    # return mp.Queue()


def get_empty_queue_exception() -> type:
    return Empty


def get_max_num_processes() -> int:
    return mp.cpu_count()


def locate_uuid(object_list: List[object], wanted_uuid: UUID):
    all_have_uuid = all([hasattr(obj, 'uuid') for obj in object_list])
    assert all_have_uuid, "Not all objects in object_list have uuid's"
    uuid_list = [obj.uuid for obj in object_list]
    if wanted_uuid not in uuid_list:
        return False, None, None
    else:
        uuid_ind = uuid_list.index(wanted_uuid)
        return True, uuid_ind, uuid_list[uuid_ind]


class PassSigFeedback:
    def __init__(self, feedback_queue: mp.JoinableQueue):
        self.fbq = feedback_queue

    def add_particlenode(self, node: DatasetTreeNode, num: int):
        self.fbq.put(ProcessSigPassTask(sig_pass_type=WorkerSigPassType.add_particlenode,
                                        sig_args=(node, num)))
        self.fbq.task_done()

    def add_all_particlenodes(self, all_nodes: list):
        self.fbq.put(ProcessSigPassTask(sig_pass_type=WorkerSigPassType.add_all_particlenodes,
                                        sig_args=all_nodes))
        self.fbq.task_done()

    def add_datasetnode(self, node: DatasetTreeNode):
        self.fbq.put(ProcessSigPassTask(sig_pass_type=WorkerSigPassType.add_datasetindex,
                                        sig_args=node))
        self.fbq.task_done()

    def reset_tree(self):
        self.fbq.put(ProcessSigPassTask(sig_pass_type=WorkerSigPassType.reset_tree))
        self.fbq.task_done()

    def bin_size(self, bin_size: int):
        self.fbq.put(ProcessSigPassTask(sig_pass_type=WorkerSigPassType.bin_size,
                                        sig_args=bin_size))
        self.fbq.task_done()

    def set_start(self, start: float):
        self.fbq.put(ProcessSigPassTask(sig_pass_type=WorkerSigPassType.set_start,
                                        sig_args=start))
        self.fbq.task_done()

    def set_tmin(self, tmin: float):
        self.fbq.put(ProcessSigPassTask(sig_pass_type=WorkerSigPassType.set_tmin,
                                        sig_args=tmin))
        self.fbq.task_done()

    def data_loaded(self):
        self.fbq.put(ProcessSigPassTask(sig_pass_type=WorkerSigPassType.data_loaded))
        self.fbq.task_done()

    def add_irf(self, decay:ndarray, time_series:ndarray, dataset:H5dataset):
        self.fbq.put(ProcessSigPassTask(sig_pass_type=WorkerSigPassType.add_irf,
                                        sig_args=(decay, time_series, dataset)))
        self.fbq.task_done()


class ProcessProgressCmd(IntEnum):
    Start = auto()
    SetMax = auto()
    AddMax = auto()
    Single = auto()
    Step = auto()
    SetValue = auto()
    Complete = auto()
    SetStatus = auto()


class ProcessProgressTask:
    def __init__(self, task_cmd: ProcessProgressCmd, args=None):
        self.task_cmd = task_cmd
        self.args = args


class ProcessProgFeedback:
    def __init__(self, feedback_queue: mp.JoinableQueue):
        self.fbq = feedback_queue

    def set_max(self, max_value: int):
        self.fbq.put(ProcessProgressTask(task_cmd=ProcessProgressCmd.SetMax, args=max_value))

    def add_max(self, max_to_add: int):
        self.fbq.put(ProcessProgressTask(task_cmd=ProcessProgressCmd.AddMax, args=max_to_add))

    def single(self):
        self.fbq.put(ProcessProgressTask(task_cmd=ProcessProgressCmd.Single))

    def step(self, value: float = None):
        if value is None:
            value = 1
        self.fbq.put(ProcessProgressTask(task_cmd=ProcessProgressCmd.Step, args=value))

    def set_value(self, value: int):
        self.fbq.put(ProcessProgressTask(task_cmd=ProcessProgressCmd.SetValue, args=value))

    def end(self):
        self.fbq.put(ProcessProgressTask(task_cmd=ProcessProgressCmd.Complete))

    def set_status(self, status):
        self.fbq.put(ProcessProgressTask(task_cmd=ProcessProgressCmd.SetStatus, args=status))

    def start(self, max_value: int = None):
        if max_value is None:
            max_value = 100
        self.fbq.put(ProcessProgressTask(task_cmd=ProcessProgressCmd.Start, args=max_value))


def prog_sig_pass(signals: ProcessThreadSignals, cmd: ProcessProgressCmd, args):
    if type(args) is not tuple:
        args = (args,)

    if cmd is ProcessProgressCmd.SetStatus:
        signals.status_update.emit(*args)
    elif cmd is ProcessProgressCmd.Start:
        signals.start_progress.emit(*args)
    elif cmd is ProcessProgressCmd.SetMax:
        signals.set_progress.emit(*args)
    elif cmd is ProcessProgressCmd.Step:
        signals.step_progress.emit(*args)
    elif cmd is ProcessProgressCmd.Complete:
        signals.end_progress.emit()
    else:
        logger.error(f"Feedback return not configured for: {cmd}")


class ProgressTracker:
    def __init__(self, num_iterations: int = None, num_trackers: int = 1):
        self.has_num_iterations = False
        self._num_iterations = None
        self._num_trackers = None
        self._step_value = None
        self._current_value = 0.0

        if num_iterations:
            self._num_iterations = num_iterations
        if num_trackers:
            self._num_trackers = num_trackers
        self.calc_step_value()

    @property
    def num_iterations(self) -> int:
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, num_iterations: int):
        assert type(num_iterations) is int, 'num_iterations is not of type int'
        self.has_num_iterations = True
        self._num_iterations = num_iterations

    @property
    def num_trackers(self) -> int:
        return self._num_trackers

    @num_trackers.setter
    def num_tracker(self, num_tracker: int):
        assert type(num_tracker) is int, 'num_tracker is not of type int'
        self._num_trackers = num_tracker

    def calc_step_value(self):
        if self._num_trackers and self.num_iterations:
            self._step_value = 100 / self._num_trackers / self._num_iterations

    def iterate(self) -> int:
        prev_value = self._current_value
        self._current_value += self._step_value
        diff_mod = self._current_value//1 - prev_value//1
        return int(diff_mod)

    def strict_iterate(self) -> float:
        prev_value = self._current_value
        self._current_value += self._step_value
        return float(self._current_value - prev_value)

    def reset(self):
        self.has_num_iterations = False
        self._num_iterations = None
        self._num_trackers = None
        self._step_value = None
        self._current_value = 0.0


class ProcessProgress(ProgressTracker):
    def __init__(self,
                 prog_fb: ProcessProgFeedback,
                 num_iterations: int = None,
                 num_of_processes: int = 1):
        super().__init__(num_iterations=num_iterations, num_trackers=num_of_processes)
        self._prog_fb = prog_fb
        self._accum_step = float(0)

    def start_progress(self):
        self._prog_fb.start(max_value=100)

    def iterate(self):
        iterate_value = super().strict_iterate()
        if self._accum_step + iterate_value >= 1.0:
            self._prog_fb.step(value=iterate_value)
            self._accum_step = 0
        else:
            self._accum_step += iterate_value


class ProcessTask:
    def __init__(self, obj: object, method_name: str, args=None):
        assert hasattr(obj, method_name), "Object does not have provided method"
        self.uuid = uuid1()
        self.obj = obj
        self.method_name = method_name
        self.args = args
    #     self.progress_queue = None
    #
    # def set_progress_queue(self, progress_queue: mp.JoinableQueue):
    #     assert type(progress_queue) is mp.queues.JoinableQueue, "process_progress incorrect type."
    #     self.progress_queue = progress_queue


class ProcessSigPassTask:
    def __init__(self, sig_pass_type: WorkerSigPassType,
                 sig_args=None):
        self.sig_pass_type = sig_pass_type
        self.sig_args = sig_args


class ProcessTaskResult:
    def __init__(self, task_uuid: UUID,
                 task_return,
                 new_task_obj: ProcessTask,
                 task_complete: bool = True):
        self.task_uuid = task_uuid
        self.task_return = task_return
        self.new_task_obj = new_task_obj
        self.task_complete = task_complete


class SingleProcess(mp.Process):
    def __init__(self, task_queue: mp.JoinableQueue,
                 result_queue: mp.JoinableQueue,
                 feedback_queue: mp.JoinableQueue = None):
        mp.Process.__init__(self)
        # assert type(task_queue) in [mp.queues.JoinableQueue, mp.managers.AutoProxy[Queue]], \
        #     'task_queue is not of type JoinableQueue'
        # assert type(result_queue) is mp.queues.JoinableQueue, \
        #     'result_queue is not of type JoinableQueue'
        # if feedback_queue:
        #     assert type(feedback_queue) is mp.queues.JoinableQueue, \
        #         'progress_queue is not of type JoinableQueue'

        self.task_queue = task_queue
        self.result_queue = result_queue
        self.feedback_queue = feedback_queue

    def run(self):
        try:
            done = False
            while not done:
                # if self.task_queue.empty():
                #     time.sleep(0.1)
                #     continue
                task = self.task_queue.get()
                if task is None:
                    done = True
                    self.task_queue.task_done()
                    self.result_queue.put(True)
                else:
                    task_run = getattr(task.obj, task.method_name)
                    if self.feedback_queue and \
                            'feedback_queue' in task_run.__func__.__code__.co_varnames:
                        if task.args is not None:
                            task_args = task.args
                            task_return = task_run(feedback_queue=self.feedback_queue, *task_args)
                        else:
                            task_return = task_run(feedback_queue=self.feedback_queue)
                    else:
                        if task.args is not None:
                            task_args = task.args
                            task_return = task_run(*task_args)
                        else:
                            task_return = task_run()
                    if task.method_name == 'run_cpa':
                        task.obj._particle = None
                        task.obj._cpa._particle = None
                    elif task.method_name == "run_grouping":
                        task.obj.particle = None
                        task.obj.best_step._particle = None
                        for step in task.obj.steps:
                            step._particle = None
                    elif task.method_name == 'fit_part_and_levels':
                        task.obj.part_hist.particle = None
                        task.obj.microtimes = None
                        for hist in task.obj.level_hists:
                            hist.particle = None
                            hist.microtimes = None
                            hist.level = None
                    process_result = ProcessTaskResult(task_uuid=task.uuid,
                                                       task_return=task_return,
                                                       new_task_obj=task.obj)
                    self.result_queue.put(process_result)
                    del task
                    if process_result.task_complete:
                        self.task_queue.task_done()
        except Exception as e:
            self.result_queue.put(e)
            # logger(e)
