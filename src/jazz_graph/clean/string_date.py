"""Tools for handling string dates.

These are aimed at known formats in discogs data, not general purpose functions.
"""
import pandas as pd

def is_year_precision(x: pd.Series) -> pd.Series:
    """True for dates dates in YYYY-MM-DD format, where MM and DD indicate unknowns."""
    is_short = x.str.len() == 4
    ends_zeros = x.str.match(r'\d\d\d\d-00-00')
    return is_short | ends_zeros

def is_month_precision(x: pd.Series) -> pd.Series:
    is_short = x.str.len() == 7
    exp = r'\d\d\d\d-(0[123456789]|[123]/d)-00'
    numeric_month = x.str.match(exp)
    return is_short | numeric_month

def is_day_precision(x: pd.Series) -> pd.Series:
    is_correct_length = x.str.len() == 10
    return is_correct_length & (~is_month_precision(x) & ~is_year_precision(x))

def date_precision(x: pd.Series) -> pd.Series:
    precision = x.copy()
    precision = precision.mask(is_year_precision(precision), 'year')
    precision = precision.mask(is_month_precision(precision), 'month')
    precision = precision.mask(is_day_precision(precision), 'day')
    return precision


def clean_string_date(x: pd.Series):
    """Convert string YYYY-MM-DD data to datetime type.

    In cases where the data represents something unknown (e.g., MM=00),
    the date is interpreted as teh earliest date compatible with what is known.

    It is recommended to use this with `date_precision` to flag cases where the
    date data represents interpolation of an unknown.
    """
    # set string lengths to 10.
    x = x.mask(x.str.len() == 4, x + '-01-01')  # pyright: ignore [reportOperatorIssue]
    x = x.mask(x.str.len() == 7, x + '-01')  # pyright: ignore [reportOperatorIssue]

    # handle ends with 00-00 and -00 cases.
    x = x.mask(x.str.endswith('00-00'), x.str[:5] + "01-01")
    x = x.mask(x.str.endswith('00'), x.str[:8] + '01')
    x = x.mask(x.str.contains('-00-'), x.str.replace('-00-', '-01-'))
    return pd.to_datetime(x)
