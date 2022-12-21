import datetime
import pprint
import time

from typing import Callable, Dict

from graph_jsp_env.disjunctive_graph_logger import log


def extrapolate_cost(callable_function: Callable, *function_args, function_kwargs: Dict = None,
                     extrapolation_limit=200, extrapolation_step=5) -> None:
    if function_kwargs is None and function_args is None:
        # no params provided
        start = time.perf_counter()
        callable_function()
        end = time.perf_counter()
    elif function_args is not None and function_kwargs is not None:
        # both provided
        start = time.perf_counter()
        callable_function(*function_args, **function_kwargs)
        end = time.perf_counter()
    elif function_kwargs is not None:
        # only kwargs provided
        start = time.perf_counter()
        callable_function(**function_kwargs)
        end = time.perf_counter()
    else:
        # only args provided
        start = time.perf_counter()
        callable_function(*function_args)
        end = time.perf_counter()

    solving_duration = end - start

    # noinspection PyUnresolvedReferences
    log.info(f"extrapolated costs for function '{callable_function.__name__}'.")
    log.info(f"function args: {pprint.pformat(function_args)}")
    log.info(f"function kwargs: {pprint.pformat(function_kwargs)}")
    log.info(f"duration of 1 run: {datetime.timedelta(seconds=int(solving_duration))}")
    for i in range(extrapolation_step, extrapolation_limit+1, extrapolation_step):
        dur = datetime.timedelta(seconds=int(i * solving_duration))
        log.info(f"cost for {i} runs: {dur}")


if __name__ == '__main__':
    extrapolate_cost(time.sleep, 1)
