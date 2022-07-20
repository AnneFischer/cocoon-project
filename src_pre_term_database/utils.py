import pandas as pd
from typing import Dict, Union, List, Optional
import json
import os
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt


DEFAULT_CORRELATION_THRESHOLD = 0.85
NOT_ASCENDING = False
numeric_type = Union[int, float, complex, np.number]


def read_settings(path: str,
                  filename: str,
                  encoding: str = 'utf-8') -> Union[pd.DataFrame, dict]:
    """Load one of the standard setting files.

    Parameters
    ----------
    path : str
        Path to the folder containing the file.
    filename : str
        Name of the setting files, without file extension.
    encoding : str, default 'utf-8'
        Encoding to use to open the files.

    Returns
    -------
    type : Union[pd.DataFrame, dict]
        Either a dataframe or a dictionary, depending on the type of file to open.
    """
    complete_filename = os.path.join(path, filename + '.json')

    with open(complete_filename, 'r', encoding=encoding) as ff:
        return json.load(ff)


def invert_mapping(mappings: dict) -> dict:
    """Invert a dictionary that maps strings to other strings, either one-to-one or one-to-many.

    For example, with an input:
    {
       'rec_id': ['RecID','RecordID']
       'abortion' : 'Abortion'
    }

    will return:
    {
       'RecID': 'rec_id',
       'RecordID': 'rec_id',
       'Abortion': 'abortion'
    }

    Parameters
    ----------
    mappings : Dict[str, Union[str, List[str]]]
        Dictionary mapping string keys to either other strings, or a list of strings.

    Returns
    -------
    type : Dict[Union[str, List[str]], str]
    """
    inverted_dict = dict()
    for key, value in mappings.items():
        if isinstance(value, list):
            for subv in value:
                inverted_dict.update({subv: key})
        else:
            inverted_dict.update({value: key})
    return inverted_dict


def rename_columns_from_mappings(
        records: pd.DataFrame,
        mappings: Dict[str, Union[str, List[str]]],
        invert_mappings: bool = True
) -> pd.DataFrame:
    """Rename the columns of a dataframe using the column names mapping.

    Parameters
    ----------
    records : pd.DataFrame
        Dataframe whose columns must be renamed.
    mappings : Dict[str, Union[str, List[str]]]
        Column names mappings as defined in the settings template. For example:
        {
            'rec_id' : 'RecID',
            'age' : 'Age'
        }
    invert_mappings : bool, optional
        Whether to invert the mappings before renaming (meaning the values become keys and vice
        versa).

    Returns
    -------
    type : pd.DataFrame
    """
    if invert_mappings:
        mappings = invert_mapping(mappings)
    return records.rename(columns=mappings)


def convert_columns_to_numeric(df: pd.DataFrame,
                               columns: List[str],
                               **kwargs):
    """Convert a column in a dataframe to numerical dtype.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe.
    columns: List[str]
        Label(s) of the column(s) to convert.
    kwargs:
        Dictionary of parameters to pass to pd.to_numeric.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with the column converted to numeric.

    """
    for column in columns:
        df.loc[:, column] = pd.to_numeric(df[column], **kwargs)

    return df


def calculate_percentage(nominator: numeric_type,
                         denominator: numeric_type) -> numeric_type:
    """"Calculate the percentage of the nominator divided by the denominator.

    Parameters
    ----------
    nominator : Union[int, float, complex, np.number]
    denominator : Union[int, float, complex, np.number]

    Returns
    -------
    type : Union[int, float, complex, np.number]
        The percentage of nominator divided by denominator rounded on two decimals.
    """

    return np.round(100 * float(nominator) / float(denominator), 2)


def replace_inf_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Replace all infinite values in df with zero.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df : pd.DataFrame
    """
    num_rows_with_infs = len(df[(df.isin([np.inf, -np.inf]).any(axis=1))])
    print(f'The number of rows with infinite numbers that will be replaced with zero is: {num_rows_with_infs}')

    df = df.replace([np.inf, -np.inf], 0)

    return df


def replace_na_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Replace all NA values in df with zero.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df : pd.DataFrame
    """
    num_rows_with_nas = len(df[(df.isnull().any(axis=1))])
    print(f'The number of rows with NA numbers that will be replaced with zero is: {num_rows_with_nas}')

    df = df.fillna(0)

    return df


def get_correlated_columns(data: pd.DataFrame,
                           cols_to_check: List[str] = None,
                           thresh: float = DEFAULT_CORRELATION_THRESHOLD
                           ) -> pd.Series:
    """Get the pairs of columns with correlation coefficient higher than a threshold.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to check
    cols_to_check : List[str], optional
        Columns to check. If None, check all columns. Default = None
    thresh : float, optional
        Only return pairs with correlation coefficient higher than this threshold. Default = 0.85

    Returns
    -------
    type: pd.Series
        Series with correlation coefficients and the pairs of columns as indices.

    """
    if cols_to_check is None:
        cols_to_check = list(data.columns)

    return (
        data.loc[:, cols_to_check]
        .corr().abs().unstack()                 # compute correlation matrix
        .loc[combinations(cols_to_check, 2)]    # only select unique pairs
        .sort_values(ascending=NOT_ASCENDING)
        .pipe(lambda f: f.loc[f > thresh])      # filter series by threshold
    )


def remove_correlated_columns(data: pd.DataFrame,
                              cols_to_check: Optional[List[str]] = None,
                              thresh: float = DEFAULT_CORRELATION_THRESHOLD
                              ) -> List[str]:
    """For each pair of columns with correlation coefficient higher than a threshold
    return only one of the two columns.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to check
    cols_to_check : List[str], optional
        Columns to check. If None, check all columns. Default = None
    thresh : float, optional
        Only return pairs with correlation coefficient higher than this threshold. Default = 0.85

    Returns
    -------
    type: List[str]
        Column list with one column removed for each pair of correlated columns

    """
    if cols_to_check is None:
        cols_to_check = list(data.columns)

    data_corr = get_correlated_columns(data=data,
                                       cols_to_check=cols_to_check,
                                       thresh=thresh)

    cols_to_remove: List[str] = []
    # loop over the columns that are highly correlated
    for row in data_corr.index:
        # if not yet one of the columns is in the remove list
        # then add one of the two columns to the remove list
        if all(row[x] not in cols_to_remove for x in [0, 1]):
            cols_to_remove.append(row[0])

    cols_filtered = list(set(cols_to_check) - set(cols_to_remove))

    return cols_filtered


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
