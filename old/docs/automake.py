import sys
import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
# from watchdog.events import LoggingEventHandler


class MyEventHandler(PatternMatchingEventHandler):

    def __init__(self, observer, *args, **kwargs):
        PatternMatchingEventHandler.__init__(self, *args, **kwargs)
        self.observer = observer

    def on_modified(self, event):
        self.observer.unschedule_all()
        os.system('make html')
        observer.schedule(self, path, recursive=True)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s - %(message)s',
    #                     datefmt='%Y-%m-%d %H:%M:%S')
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    # event_handler = LoggingEventHandler()
    observer = Observer()
    event_handler = MyEventHandler(observer, patterns='*.rst')
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()