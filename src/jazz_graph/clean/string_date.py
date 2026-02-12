"""Tools for handling string dates.

These are aimed at known formats in discogs data, not general purpose functions.
"""
import pandas as pd

def _is_year_precision(x: pd.Series) -> pd.Series:
    """True for dates dates in YYYY-MM-DD format, where MM and DD indicate unknowns."""
    is_short = x.str.len() == 4
    ends_zeros = x.str.match(r'\d\d\d\d-00-00')
    short_ends_zeros = (x.str.len() == 7) & x.str.endswith('-00')
    return is_short | ends_zeros | short_ends_zeros

def _is_month_precision(x: pd.Series) -> pd.Series:
    is_short = x.str.len() == 7
    exp = r'\d\d\d\d-(0[123456789]|[123]/d)-00'
    numeric_month = x.str.match(exp)
    return (is_short & (~x.str.endswith('-00'))) | numeric_month

def _is_day_precision(x: pd.Series) -> pd.Series:
    is_correct_length = x.str.len() == 10
    return is_correct_length & (~_is_month_precision(x) & ~_is_year_precision(x))

def date_precision(x: pd.Series) -> pd.Series:
    precision = x.copy()
    precision = precision.mask(_is_year_precision(precision), 'year')
    precision = precision.mask(_is_month_precision(precision), 'month')
    precision = precision.mask(_is_day_precision(precision), 'day')
    return precision

def _convert_year_precision(x: pd.Series):
    x = x.str.replace('-00', '')
    return pd.to_datetime(x, format='mixed') + pd.DateOffset(years=1) - pd.DateOffset(days=1)

def _convert_month_precision(x: pd.Series):
    x = x.str.replace('-00', '')
    x = pd.to_datetime(x, format='mixed') + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    return x

def _convert_day_presion(x: pd.Series):
    x = x.str.replace('-00', '')
    x = pd.to_datetime(x, format='mixed')
    return x

def clean_string_date(x: pd.Series):
    """Convert string YYYY-MM-DD data to datetime type.

    In cases where the data represents something unknown (e.g., MM=00),
    the date is interpreted as teh earliest date compatible with what is known.

    It is recommended to use this with `date_precision` to flag cases where the
    date data represents interpolation of an unknown.
    """
    precision = date_precision(x)
    year = x.mask(precision == 'year', _convert_year_precision(x))
    month = x.mask(precision == 'month', _convert_month_precision(x))  # pyright: ignore [reportOperatorIssue]
    day = x.mask(precision == 'day', _convert_day_presion(x))
    x = x.mask(precision == 'year', year)
    x = x.mask(precision == 'month', month)
    x = x.mask(precision == 'day', day)
    return x
