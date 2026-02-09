import pandas as pd
from jazz_graph.clean.string_date import clean_string_date, is_month_precision, date_precision

def test_is_month_precision():
    x = pd.Series([
        '2010-01-03', '2010', '2010-01', '2010-00-00', '2010-03-00'
    ])
    result = is_month_precision(x)
    assert (result == pd.Series([False, False, True, False, True])).all(), result

def test_date_precision():
    x = pd.Series([
        '2010-01-03', '2010', '2010-01', '2010-00-00', '2010-03-00'
    ])
    expected = pd.Series(['day', 'year', 'month', 'year', 'month'])
    assert (date_precision(x) == expected).all(), date_precision(x)

def test_clean_string_date():
    x = pd.Series([
        '2010-01-03', '2011', '2012-01', '2013-00-00', '2014-03-00', '2000-00'
    ])
    expected = pd.to_datetime(pd.Series([
        '2010-01-03', '2011-01-01', '2012-01-01', '2013-01-01', '2014-03-01', '2000-01-01'
    ]))
    assert (clean_string_date(x) == expected).all(), clean_string_date(x)