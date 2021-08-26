import timeit, datetime
from functools import wraps


def log_step(funct):
    @wraps(funct)
    def wrapper(*args, **kwargs):
        tic = timeit.default_timer()
        result = funct(*args, **kwargs)
        time_taken = datetime.timedelta(seconds=timeit.default_timer() - tic)
        print(f"Just ran '{funct.__name__}' function. Took: {time_taken}")
        return result
    return wrapper