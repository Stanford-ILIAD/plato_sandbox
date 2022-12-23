"""
Logging is very important. It helps you:
- look at old experiments and see what happened
- track down bugs
- monitor ongoing experiments
and many other things.

My current favorite is loguru https://loguru.readthedocs.io/en/stable/index.html
"""
from loguru import logger


def setup(log_fname=None):
    if log_fname is not None:
        logger.add(log_fname)


def debug(s):
    logger.opt(depth=1).debug(s)


def info(s):
    logger.opt(depth=1).info(s)


def warn(s):
    logger.opt(depth=1).warning(s)
