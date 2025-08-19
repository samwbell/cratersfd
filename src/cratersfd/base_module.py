import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from scipy.stats.mstats import gmean
from scipy.stats import gamma, poisson, linregress, norm, skew, lognorm
from scipy.special import gammainc, lambertw
from scipy.special import gamma as gf
from scipy import optimize
from scipy.optimize import root_scalar, minimize
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.signal import fftconvolve, savgol_filter
import importlib.resources as ior
import pickle as pkl
import ash
import time
import random


def format_runtime(seconds, round_to=5):
    days = math.floor(seconds / (60 * 60 * 24))
    seconds_in_final_day = seconds - days * (60 * 60 * 24)
    hours = math.floor(seconds_in_final_day / (60 * 60))
    seconds_in_final_hour = seconds_in_final_day - hours * (60 * 60)
    minutes = math.floor(seconds_in_final_hour / 60)
    seconds_in_final_minute = seconds_in_final_hour - minutes * 60
    return_string = str(round(seconds_in_final_minute, round_to)) + ' seconds'
    if minutes != 0:
        return_string =  str(minutes) + ' minutes, ' + return_string
    if hours != 0:
        return_string =  str(hours) + ' hours, ' + return_string
    if days != 0:
        return_string =  str(days) + ' days, ' + return_string
    return return_string

def runtime(t1, t2):
    print(format_runtime(t2 - t1))


def get_kwargs(func):
    defaults = func.__defaults__
    if not defaults:
        return {}

    n_args = func.__code__.co_argcount
    var_names = func.__code__.co_varnames
    kwarg_names = var_names[:n_args][-len(defaults):]
    kwargs = {kwarg_name: default for kwarg_name, 
                   default in zip(kwarg_names, defaults)}
    return kwargs


def get_arg_names(func):
    defaults = func.__defaults__
    if not defaults:
        return {}

    n_args = func.__code__.co_argcount
    var_names = func.__code__.co_varnames
    return var_names[:n_args]


def match_args(_locals, func, exclude=None):
    args = {
        k : v 
        for k, v in _locals.items()
        if k in get_arg_names(func)
    }
    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        for key in exclude:
            args.pop(key)
    return args


def get_kwarg_names(func):
    defaults = func.__defaults__
    if not defaults:
        return {}

    n_args = func.__code__.co_argcount
    var_names = func.__code__.co_varnames
    return var_names[:n_args][-len(defaults):]


def match_kwargs(_locals, func, exclude=None):
    kwargs = {
        k : v 
        for k, v in _locals.items()
        if k in get_kwarg_names(func)
    }
    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        for key in exclude:
            kwargs.pop(key)
    return kwargs


def is_equally_spaced(X, tolerance=1E-3):
    spacing = np.diff(X)
    mismatch = np.abs(spacing / spacing.mean() - 1)
    return bool(np.sum(mismatch > tolerance) < 1)

                                                     
def scientific_notation(n, rounding_n=2):
    e = math.floor(np.log10(n))
    n_str = str(round(n / 10**e, rounding_n))
    return n_str + f"x10$^{{{e}}}$"


cs = [
    'xkcd:dark sky blue', 'xkcd:vivid blue', 'xkcd:royal blue', 
    'xkcd:light orange', 'xkcd:orange', 'xkcd:dark orange', 
    'xkcd:pink', 'xkcd:pink red', 'xkcd:bright red',
    'xkcd:turquoise', 'xkcd:slime green', 'xkcd:teal green'
]

