# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Callable


def call_once(func: Callable) -> Callable:
    """Decorate a function such that it is only called once

    Examples:

    .. code-block:: python

       @cufile_python.utils.call_once
       foo(args)

    Parameters
    ----------
    func: Callable
        The function to be decorated.

    Returns
    -------
    Callable
        A decorated function.
    """
    once_flag = True
    cached_result = None

    def wrapper(*args, **kwargs):
        nonlocal once_flag
        nonlocal cached_result
        if once_flag:
            once_flag = False
            cached_result = func(*args, **kwargs)
        return cached_result

    return wrapper


def kvikio_deprecation_notice(msg: str = "This function is deprecated.", *, since: str) -> Callable:
    """Decorate a function to print the deprecation notice at runtime,
       and also add the notice to Sphinx documentation.

    Examples:

    .. code-block:: python

       @kvikio.utils.kvikio_deprecation_notice("Use bar(args) instead.")
       foo(args)

    Parameters
    ----------
    msg: str
        The deprecation notice.

    since: str
        The KvikIO version since which the function becomes deprecated.

    Returns
    -------
    Callable
        A decorated function.
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            warnings.warn(msg, category=FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        # Allow the docstring to be correctly generated for the decorated func in Sphinx
        func_doc = getattr(func, "__doc__")
        valid_docstring = "" if func_doc is None else func_doc
        wrapper.__doc__ = "{:} {:} {:}\n\n{:}".format(".. deprecated::", since, msg, valid_docstring)

        return wrapper

    return decorator


def kvikio_deprecate_module(msg: str = "", *, since: str) -> None:
    """Mark a module as deprecated.

    Parameters
    ----------
    msg: str
        The deprecation notice.

    since: str
        The KvikIO version since which the module becomes deprecated.
    """
    full_msg = f"This module is deprecated since {since}. {msg}"
    warnings.warn(full_msg, category=FutureWarning, stacklevel=2)
