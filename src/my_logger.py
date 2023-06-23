from __future__ import annotations
import logging
import sys
from typing import Union


def setup_logger(
    logger: Union[str, logging.Logger], is_main: bool = None
) -> logging.Logger:
    assert type(logger) in [str, logging.Logger], "Provided logger not " "correct type"

    class StdOutFilter(logging.Filter):
        def filter(self, record: logging.LogRecord):
            return record.levelno in (logging.DEBUG, logging.INFO, logging.WARNING)

    class StdErrFilter(logging.Filter):
        def filter(self, record: logging.LogRecord):
            return record.levelno not in (logging.DEBUG, logging.INFO, logging.WARNING)

    if type(logger) is str:
        logger = logging.getLogger(logger)

    if is_main:
        logger.root.handlers = []

    line = "-" * 100
    err_fmt = (
        f"{line}\n%(asctime)s -> %(threadName)s -> %(module)s -> %(funcName)s(%(lineno)d): "
        f"%(message)s\n{line}"
    )
    err_formatter = logging.Formatter(fmt=err_fmt)

    debug_fmt = (
        f"%(asctime)s -> %(threadName)s -> %(module)s -> %(funcName)s: " f"%(message)s"
    )
    debug_formatter = logging.Formatter(fmt=debug_fmt)

    cli_err = logging.StreamHandler(stream=sys.stderr)
    cli_err.setLevel(logging.ERROR)
    cli_err.setFormatter(err_formatter)
    cli_err.addFilter(StdErrFilter())

    cli_out = logging.StreamHandler(stream=sys.stdout)
    cli_out.setLevel(logging.INFO)
    cli_out.setFormatter(debug_formatter)
    cli_out.addFilter(StdOutFilter())

    file_log = logging.FileHandler(filename="log.txt", mode="w+")
    file_log.setLevel(logging.INFO)
    file_log.setFormatter(err_formatter)

    root_logger = logging.root
    root_logger.setLevel(logging.ERROR)
    root_logger.addHandler(cli_err)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(cli_out)

    num_args = len(sys.argv) - 1
    if num_args == 0 or (num_args > 0 and "--dev" not in sys.argv):
        root_logger.addHandler(file_log)
    if num_args and ("--dev" in sys.argv or "--debug" in sys.argv):
        cli_out.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    return logger
