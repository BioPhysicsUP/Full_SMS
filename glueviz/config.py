from glue.config import data_factory
import pandas as pd


def is_parquet(filename: str, **kwargs) -> bool:
    with open(filename, 'rb') as file:
        magic_number = file.read(4)
    return magic_number == b'PAR1'


@data_factory('Parquet File', is_parquet)
def read_parquet(file_name):
    return pd.read_parquet(file_name)
