import os
from pathlib import Path
import time
import uuid
import warnings
from functools import wraps
from typing import Callable, Literal

import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image

DEBUG = True

def ftimed(func=None):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not DEBUG:
                return func(*args, **kwargs)
            else:
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                print(f"Time taken by {func.__qualname__}: {end_time - start_time} seconds")
                return result
        return wrapper


    if func is None:
        return decorator
    else:
        return decorator(func)


class ctimed:
    """
    Context manager for timing lines of code. Use like:
    ```
    with ctimed(name="Model Forward"):
        y = model(x)
    ```
    """
    def __init__(self, name=None):
        self.name = name
        self.start_time = None

    def __enter__(self):
        if DEBUG:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if DEBUG:
            end_time = time.perf_counter()
            if self.name:
                print(f"Time taken by {self.name}: {end_time - self.start_time} seconds")
            else:
                print(f"Time taken: {end_time - self.start_time} seconds")


def print_gpu_memory(clear_mem: Literal["pre", "post", None] = "pre"):
    if not torch.cuda.is_available():
        warnings.warn("Warning: CUDA device not available. Running on CPU.")
        return
    if clear_mem == "pre":
        torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"Memory allocated: {allocated / (1024**2):.2f} MB")
    print(f"Memory reserved: {reserved / (1024**2):.2f} MB")
    print(f"Total memory: {total / (1024**2):.2f} MB")
    if clear_mem == "post":
        torch.cuda.empty_cache()

def cuda_empty_cache(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    return wrapper

def print_first_param(module):
    print(list(module.parameters())[0])

def fdebug(func=None, *, exclude=None):
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            arg_vals = args[:len(arg_names)]
            arg_vals = [
                (str(value)+str(value.shape) if isinstance(value, torch.Tensor) else value)
                for value in arg_vals
            ]
            args_pairs = ", ".join(f"{name}={value}" for name, value in zip(arg_names, arg_vals) if name not in exclude)
            kwargs_pairs = ", ".join(f"{k}={v}" for k, v in kwargs.items() if k not in exclude)
            all_args = ", ".join(filter(None, [args_pairs, kwargs_pairs]))
            print(f"Calling {func.__name__}({all_args})")
            result = func(*args, **kwargs)
            print(f"{func.__name__} returned {str(result)+str(result.shape) if isinstance(result, torch.Tensor) else result}")
            return result
        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)

class IncrementIndex:
    def __init__(self, max:int=100):
        self.retry_max = max
        self.retries = 0

    def __call__(self, index):
        if self.retries > self.retry_max:
            raise RuntimeError(f"Retried too many times, max:{self.retry_max}")
        else:
            self.retries += 1
        index += 1
        return index

_identity = lambda x: x

def fretry(func=None, *, exceptions=(Exception,), mod_args:tuple[Callable|None, ...]=tuple(), mod_kwargs:dict[str,Callable|None]=dict()):
    def decorator(func):
        @wraps(func)
        def fretry_wrapper(*args, **kwargs):
            try:
                out = func(*args, **kwargs)
            except exceptions as e:
                new_args = []
                for i, arg in enumerate(args):
                    if i < len(mod_args):
                        mod_func = mod_args[i] or _identity
                        new_args.append(mod_func(arg))
                    else:
                        new_args.append(arg)
                new_kwargs = {}
                for k, kwarg in kwargs.items():
                    if k in mod_kwargs:
                        mod_func = mod_kwargs[k] or _identity
                        new_kwargs[k] = mod_func(kwarg)
                kwargs.update(new_kwargs)

                import traceback
                traceback.print_exc()
                warnings.warn(
                    f"Function {func} failed due to {e} with inputs {args}, {kwargs}, "
                    f"retrying with modified inputs {new_args}, {new_kwargs}"
                )
                out = fretry_wrapper(*new_args, **new_kwargs)
            return out
        return fretry_wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def texam(t: torch.Tensor):
    print(f"Shape: {tuple(t.shape)}")
    if t.dtype.is_floating_point or t.dtype.is_complex:
        mean_val = t.mean().item()
    else:
        mean_val = "N/A"
    print(f"Min: {t.min().item()}, Max: {t.max().item()}, Mean: {mean_val}")
    print(f"Device: {t.device}, Dtype: {t.dtype}, Requires Grad: {t.requires_grad}")