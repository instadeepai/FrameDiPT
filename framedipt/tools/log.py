"""Module for providing a configured logger."""
from __future__ import annotations

import os

import neptune
import omegaconf
from absl import logging


def get_logger() -> logging.ABSLLogger:
    """Configuring absl logging.

    Returns:
        logger: configured logger.
    """
    logging.use_absl_handler()
    logging.set_verbosity(logging.INFO)
    return logging.get_absl_logger()
