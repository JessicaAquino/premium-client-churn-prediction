import logging
import functools
import time

def process_log(func):
    logger = logging.getLogger(func.__module__)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Input
        if args and hasattr(args[0], "shape"):
            rows, cols = args[0].shape
            logger.info(f"Starting {func.__name__} with shape {rows} rows x {cols} cols")
        else:
            logger.info(f"Starting {func.__name__}")

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        #Output
        try:
            if hasattr(result, 'shape'):
                rows, cols = result.shape
                logger.info(f"Finished {func.__name__} with shape {rows} rows x {cols} cols. Exec time: {elapsed:.2f} s")
            else:
                logger.info(f"Finished {func.__name__}. Result type: {type(result)}. Exec time: {elapsed:.2f} s")
        except Exception as e:
            logger.warning(f"Error logging shape of result in {func.__name__}: {e}")

        return result
    
    return wrapper