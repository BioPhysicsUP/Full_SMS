import multiprocessing
import os
import time

NUM_PROCESSES = 20
NUM_QUEUE_ITEMS = 20  # so really 40, because hello and world are processed separately


def worker_main(queue, result_queue):
    print(os.getpid(), "working")
    while True:
        item = queue.get(
            block=True)  # block=True means make a blocking call to wait for items in queue
        result_queue.put('done')
        if item is None:
            break

        print(os.getpid(), "got", item)
        time.sleep(1)  # simulate a "long" operation


def main():
    the_queue = multiprocessing.Queue()
    the_result_queue = multiprocessing.Queue()
    the_pool = multiprocessing.Pool(NUM_PROCESSES, worker_main, (the_queue, the_result_queue))

    for i in range(NUM_QUEUE_ITEMS):
        the_queue.put("hello")
        the_queue.put("world")

    while not the_result_queue.empty():
        print(the_result_queue.get())

    for i in range(NUM_PROCESSES):
        the_queue.put(None)

    # prevent adding anything more to the queue and wait for queue to empty
    the_queue.close()
    the_queue.join_thread()

    # prevent adding anything more to the process pool and wait for all processes to finish
    the_pool.close()
    the_pool.join()


if __name__ == '__main__':
    main()