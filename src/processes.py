import multiprocessing as mp
from enum import IntEnum, auto
from queue import Empty
from typing import List, Union
from uuid import UUID, uuid1

from my_logger import setup_logger

logger = setup_logger(__name__)


def create_queue() -> mp.JoinableQueue:
    return mp.JoinableQueue()


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


class ProcessTask:
    def __init__(self, obj: object, method_name: str):
        assert hasattr(obj, method_name), "Object does not have provided " \
                                          "method"
        self.uuid = uuid1()
        self.obj = obj
        self.method_name = method_name


class ProcessTaskResult:
    def __init__(self, task_uuid: UUID,
                 task_return,
                 new_task_obj: ProcessTask):
        self.task_uuid = task_uuid
        self.task_return = task_return
        self.new_task_obj = new_task_obj


class ProcessProgressCmd(IntEnum):
    SetMax = auto()
    AddMax = auto()
    Single = auto()
    Step = auto()
    SetValue = auto()
    Complete = auto()


class ProcessProgressTask:
    def __init__(self, task_cmd: ProcessProgressCmd, value: int):
        self.task_cmd = task_cmd
        self.value = value


class ProgressTracker:
    def __init__(self, num_iterations: int = None, num_trackers: int = 1):
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


class ProcessProgress(ProgressTracker):
    def __init__(self, progress_queue: Union[mp.Queue, mp.JoinableQueue],
                 num_iterations: int = None, num_of_processes: int = 1):
        super().__init__(num_iterations=num_iterations, num_trackers=num_of_processes)
        self.progress_queue = progress_queue

    def iterate(self):
        iterate_value = super().iterate()
        if iterate_value:
            task = ProcessProgressTask(task_cmd=ProcessProgressCmd.Step, value=iterate_value)
            self.progress_queue.put(task)


class SingleProcess(mp.Process):
    def __init__(self, task_queue: mp.JoinableQueue,
                 result_queue: mp.JoinableQueue,
                 progress_queue: mp.JoinableQueue = None):
        mp.Process.__init__(self)
        assert type(task_queue) is mp.queues.JoinableQueue, \
            'task_queue is not of type JoinableQueue'
        assert type(result_queue) is mp.queues.JoinableQueue, \
            'result_queue is not of type JoinableQueue'
        if progress_queue:
            assert type(progress_queue) is mp.queues.JoinableQueue, \
                'progress_queue is not of type JoinableQueue'

        self.task_queue = task_queue
        self.result_queue = result_queue
        self.progress_queue = progress_queue

    def run(self):
        try:
            done = False
            while not done:
                task = self.task_queue.get()
                if task is None:
                    done = True
                    self.task_queue.task_done()
                    self.result_queue.put(True)
                else:
                    task_run = getattr(task.obj, task.method_name)
                    if self.progress_queue and \
                            'progress_queue' in task_run.__func__.__code__.co_varnames:
                        task_return = task_run(progress_queue=self.progress_queue)
                    else:
                        task_return = task_run()
                    process_result = ProcessTaskResult(task_uuid=task.uuid,
                                                       task_return=task_return,
                                                       new_task_obj=task.obj)
                    self.result_queue.put(process_result)
                    self.task_queue.task_done()
        except Exception as e:
            self.result_queue.put(e)
