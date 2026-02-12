import pandas as pd
from jazz_graph.clean.string_date import clean_string_date, _is_year_precision, _is_month_precision, date_precision

def test_is_month_precision():
    x = pd.Series([
        '2010-01-03', '2010', '2010-01', '2010-00-00', '2010-03-00', '2000-00'
    ])
    result = _is_month_precision(x)
    assert (result == pd.Series([False, False, True, False, True, False])).all(), result

def test_is_year_precision():
    x = pd.Series([
        '2010-01-03', '2010', '2010-01', '2010-00-00', '2010-03-00', '2000-00'
    ])
    result = _is_year_precision(x)
    assert (result == pd.Series([False, True, False, True, False, True])).all(), result.values

def test_date_precision():
    x = pd.Series([
        '2010-01-03', '2010', '2010-01', '2010-00-00', '2010-03-00', '2000-00'
    ])
    expected = pd.Series(['day', 'year', 'month', 'year', 'month', 'year'])
    assert (date_precision(x) == expected).all(), date_precision(x)

def test_clean_string_date():
    x = pd.Series([
        '2010-01-03', '2011', '2012-01',
        '2013-00-00', '2014-03-00', '2000-00',
        '2010-11'
    ])
    expected = pd.to_datetime(pd.Series([
        '2010-01-03', '2011-12-31', '2012-01-31',
        '2013-12-31', '2014-03-31', '2000-12-31',
        '2010-11-30'
    ]))
    assert (clean_string_date(x) == expected).all(), clean_string_date(x)