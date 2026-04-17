from functools import wraps
import time


# --- Decorators ---

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def retry(times=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == times:
                        raise
                    print(f"Attempt {attempt} failed: {e}. Retrying...")
        return wrapper
    return decorator


# --- Generators ---

def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def chunked(iterable, size):
    iterable = list(iterable)
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def read_in_batches(data, batch_size=3):
    yield from chunked(data, batch_size)


# --- Lambda & higher-order functions ---

double = lambda x: x * 2
is_even = lambda x: x % 2 == 0

def apply(func, values):
    return list(map(func, values))

def keep(predicate, values):
    return list(filter(predicate, values))

def pipeline(*funcs):
    def execute(value):
        for f in funcs:
            value = f(value)
        return value
    return execute


# --- Context manager ---

class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        print(f"Elapsed: {self.elapsed:.4f}s")


# --- Demo ---

@timer
def slow_sum(n):
    return sum(range(n))


@retry(times=3)
def flaky(attempt_count=[0]):
    attempt_count[0] += 1
    if attempt_count[0] < 3:
        raise ValueError("not ready yet")
    return "success"


if __name__ == "__main__":
    print("=== Timer decorator ===")
    result = slow_sum(1_000_000)
    print(f"Result: {result}")

    print("\n=== Retry decorator ===")
    print(flaky())

    print("\n=== Fibonacci generator (first 10) ===")
    gen = fibonacci()
    print([next(gen) for _ in range(10)])

    print("\n=== Chunked generator ===")
    for batch in read_in_batches(range(10), batch_size=3):
        print(batch)

    print("\n=== Lambda & higher-order functions ===")
    nums = list(range(10))
    print("Doubled:", apply(double, nums))
    print("Evens:  ", keep(is_even, nums))

    print("\n=== Pipeline ===")
    process = pipeline(lambda x: x * 2, lambda x: x + 1, lambda x: x ** 2)
    print("pipeline(5):", process(5))

    print("\n=== Context manager ===")
    with Timer():
        total = sum(range(500_000))
    print(f"Sum: {total}")
