import time


def wait(secs):
    """wait decorator"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            time.sleep(secs)
            return func(*args, **kwargs)

        return wrapper

    return decorator
