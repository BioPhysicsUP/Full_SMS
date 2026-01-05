# Parquet Extension for GlueViz
This is to add the ability for GlueViz to open and load Parquet files.

This process can also be accessed via a GitHub Gist [here](https://gist.github.com/JoshuaBotha/771d555464417bef6f8f23b50a897323).

## Requirements
- GlueViz needs to be installed. This can be done in the following ways:
  - pip: With this command `pip install glueviz[all,qt]`
  - conda: With this command `conda install -c glueviz glueviz`
  - Anaconda: Once installed, GlueViz can be installed via the Anaconda Navigator
  - Standalone: By following the instructions at this [link](http://docs.glueviz.org/en/stable/installation/standalone.html)
- The Pandas and pyarrow packages need to be installed in the same environment as GlueViz
  - pip: With this command `pip install pandas pyarrow`

## Instructions
A file called `config.py` needs to be added to the `.glue` folder in the home directory.
The contents of this file shoul be:
```python
from glue.config import data_factory
import pandas as pd


def is_parquet(filename: str, **kwargs) -> bool:
    with open(filename, 'rb') as file:
        magic_number = file.read(4)
    return magic_number == b'PAR1'


@data_factory('Parquet File', is_parquet)
def read_parquet(file_name):
    return pd.read_parquet(file_name)
```

Reload GlueViz if it was open, and there should be a new filetype when loading file.