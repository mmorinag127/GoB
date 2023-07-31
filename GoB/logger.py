import logging
import sys

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    green = "\x1b[32;20m"
    blue  = "\x1b[34;20m"
    red = "\x1b[31;20m"
    reset = "\x1b[0m"
    base = grey + "%(asctime)s: " + blue + "%(name)12s " + reset 
    fmt =  "[%(levelname)5s] %(message)s "    
    
    FORMATS = {
        logging.DEBUG:    base + grey   + fmt + reset,
        logging.INFO:     base + green  + fmt + reset,
        logging.WARNING:  base + yellow + fmt + reset,
        logging.ERROR:    base + red    + fmt + reset,
        logging.CRITICAL: base + red    + fmt + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S') #'%Y/%m/%d %I:%M:%S'
        return formatter.format(record)

loggers = {}

def make_logger(name, level = 'DEBUG'):
    if 'DEBUG' in level : 
        lvl = logging.DEBUG
    elif 'INFO' in level : 
        lvl = logging.INFO
    elif 'WARNING' in level : 
        lvl = logging.WARNING
    elif 'ERROR' in level : 
        lvl = logging.ERROR
    elif 'CRITICAL' in level : 
        lvl = logging.CRITICAL
    else:
        raise ValueError(f'make_logger : {level} is not supported...')
    
    global loggers

    if loggers.get(name):
        return loggers.get(name)
    
    logger = logging.getLogger(name)
    logger.setLevel(lvl)
    
    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(lvl)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    loggers[name] = logger
    logger.propagate = False
    
    return logger
