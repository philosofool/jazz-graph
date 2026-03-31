import pandas as pd
from numpy.typing import ArrayLike


def map_array(arr: ArrayLike, mapping: dict):
    """Map the values in a 1d-array."""
    series = pd.Series(arr)  # pyright: ignore [reportCallIssue, reportArgumentType]
    return series.map(mapping).values


def map_by_index(arr):
    """Create a dictionary mapping values in arr to their index in arr.

    Raises a value error if elements of arr are not unique.
    """
    out = {}
    for i, v in enumerate(arr):
        if v in out:
            raise ValueError("The values in arr must be unique.")
        out[v] = i
    return out