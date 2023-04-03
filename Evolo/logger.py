# -*- coding: utf-8 -*-<模块功能已了解>
import logging.config

"""
配置logging基本的设置，然后在控制台输出日志，
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")
"""


def configure_logging():
    DEFAULT_LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "basic": {
                "format": "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            }
        },
        "handlers": {
            "console": {
                "formatter": "basic",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            }
        },
        "loggers": {
            "Evolo": {"handlers": ["console"], "level": "DEBUG"},
        },
    }
    logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)


def get_logger(module):
    return logging.getLogger(module)
