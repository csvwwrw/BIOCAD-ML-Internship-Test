# -*- coding: utf-8 -*-
'''
Centralized logging configuration.

Sets up file + console logging with proper formatting.
All modules use: _logger = logging.getLogger(__name__)

Created on Thu Jan 29 19:33:09 2026

@author: csvwwrw
'''

import logging
import sys
from pathlib import Path
from config.config import Config

def setup_logging(
        log_level=Config.LOG_LEVEL,
        log_file=str(Config.LOG_FILE),
        console=True
        ):
    '''
    Configure logging for entire RAG system.
    
    Creates unified logger with file + console handlers. All module loggers
    inherit this configuration automatically via root logger.
    
    Args:
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        
        log_file (str): Path to log file. Creates parent dirs if needed.
        
        console (bool): Enable console output. Defaults to True.
    
    Returns:
        None
    '''
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    root_logger.handlers.clear()
    
    formatter = logging.Formatter(
        fmt=Config.LOG_FORMAT,
        datefmt=Config.LOG_DATE_FORMAT
        )
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('faiss').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    root_logger.info(f'Logging configured: {log_file} (level={log_level})')


def get_module_logger(name):
    '''
    Get logger for specific module.
    
    Convenience function, equivalent to logging.getLogger(__name__).
    
    Args:
        name (str): Module name (use __name__).
    
    Returns:
        logging.Logger: Configured logger instance.
    '''
    return logging.getLogger(name)


if __name__ == '__main__':
    # Test logging setup
    setup_logging('DEBUG', 'logs/test.log')
    
    logger = logging.getLogger(__name__)
    logger.debug('Debug message')
    logger.info('Info message')
    logger.warning('Warning message')
    logger.error('Error message')
    
    print('Check logs/test.log for output')
