"""Small builders shared by the tests."""
import pandas as pd

from src.data_loader import SAMPLE_PATH, read_csv_content
from src.validation import validate_and_normalize


def frame(rows, indicators=("ind",)):
    """rows: iterable of (district, month, *values)."""
    cols = ["district", "month", *indicators]
    return pd.DataFrame(list(rows), columns=cols)


def sample_clean():
    load = read_csv_content(SAMPLE_PATH.read_bytes())
    clean, report = validate_and_normalize(load.raw, load.header_duplicates)
    return clean, report


def csv_clean(text, **kw):
    load = read_csv_content(text)
    return validate_and_normalize(load.raw, load.header_duplicates, **kw)
