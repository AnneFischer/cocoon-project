from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Union, Dict
from sklearn.model_selection import train_test_split
import constants as c
import math
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, ConcatDataset
import antropy as ant
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from utils import remove_correlated_columns, replace_na_with_zero, replace_inf_with_zero
import random
from src_pre_term_database.load_dataset import build_demographics_dataframe


def butter_bandpass_filter(data: np.array, low_cut: float, high_cut: float,
                           fs: float, order: int, axis: int = 0):
    """Apply a backward and forward bandpass butterworth filter with zero-phase filtering.

    The filter using Scipy package does not yield the exact same result as when using Matlab's function:

    https://stackoverflow.com/questions/55265218/building-a-filter-with-python-matlab-results-are-not-the-same
    https://dsp.stackexchange.com/questions/11466/differences-between-python-and-matlab-filtfilt-function

    Parameters
    ----------
    data : np.array
    low_cut : float
        Lower bound cutoff for high pass filter.
    high_cut : float
        Upper bound cutoff for low pass filter.
    fs : float
        Sampling frequency in Hz
    order : int
        Filter order for butterworth bandpass.
    axis : int
        Axis to perform filtering.

    Returns
    -------
        Bandpass filtered data.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low_cut / nyq, high_cut / nyq], btype='band')

    return filtfilt(b, a, data, axis=axis)


def create_filtered_channels(df_signals: pd.DataFrame, list_of_channels: List[str],
                             list_of_bandwidths: List[List[np.array]],
                             fs: int = 20, order: int = 4) -> pd.DataFrame:
    """Create filtered channels based on the Butterworth filtering scheme.

    Parameters
    ----------
    df_signals : pd.DataFrame
        Original signal data.
    list_of_channels : List[str]
        List with the names of the channels you want to filter.
    list_of_bandwidths : List[List[np.array]]
        List containing the bandwidths (low cut and high cut) for which you want to filter.
        Example: [[0.08, 4], [0.3, 3]].
    fs : int
        Sampling frequency in Hz.
    order : int
        Order of the Butterworth filter.

    Returns
    -------
    df_signals_new : pd.DataFrame
        Dataframe that contains both the original signals as the filtered signals of each channel.
    """
    # The filtered signals will be stored in df_signals_new and this df will be returned
    df_signals_new = pd.DataFrame()
    df_filtered_signals = pd.DataFrame(columns=[c.REC_ID_NAME] + list_of_channels)

    for rec_id in df_signals[c.REC_ID_NAME].unique():
        rec_id = int(rec_id)

        df_tmp_rec_id = df_signals[df_signals[c.REC_ID_NAME] == rec_id].copy()
        df_filtered_signals = pd.DataFrame(df_tmp_rec_id[[c.REC_ID_NAME]], columns=[c.REC_ID_NAME])

        for channel in list_of_channels:
            # The original (unfiltered) signal data is also added to the final dataframe
            df_filtered_signals = pd.concat([df_filtered_signals, df_tmp_rec_id[[f'{channel}']]], axis=1)

            for bandwidth in list_of_bandwidths:
                # We filter the signal data for each given bandwidth
                filtered_signals = butter_bandpass_filter(df_tmp_rec_id[f'{channel}'], bandwidth[0], bandwidth[1], fs,
                                                          order, axis=0)

                index_tmp = df_tmp_rec_id[[c.REC_ID_NAME]].index

                df_filtered_signals = pd.concat(
                    [df_filtered_signals, pd.DataFrame(filtered_signals,
                                                       columns=[f'{channel}_filt_{bandwidth[0]}_{bandwidth[1]}_hz'],
                                                       index=index_tmp)], axis=1)

        df_signals_new = pd.concat([df_signals_new, df_filtered_signals], ignore_index=True)

    return df_signals_new


def remove_first_n_samples_of_signals(df_signals: pd.DataFrame, n: int = 3600) -> pd.DataFrame:
    """Remove the first n samples per rec_id because of transient effects of the filtering.

    When using filtered channels, note that the first and last 180 seconds of the signals should be ignored since
    these intervals contain transient effects of the filters: https://physionet.org/content/tpehgdb/1.0.1/#ref1

    Parameters
    ----------
    df_signals : pd.DataFrame
        Filtered signal data.
    n : int
        The number of first samples you want to remove from the filtered data.

    Returns
    -------
    df_signals_new : pd.DataFrame
        Dataframe that contains the signals with the first n samples removed (per rec_id)
    """
    # We add a column that accumulates the number of time steps for each rec_id, such that we can remove the first
    # n samples per rec id
    df_signals.loc[:, 'time_step'] = df_signals.groupby(c.REC_ID_NAME).cumcount()

    print(f'The number of data points before removing the first {n} data points (per rec_id) is: {len(df_signals)}')

    df_signals_new = df_signals[(df_signals['time_step'] >= n)].reset_index(drop=True).copy()

    print(f'The number of data points after removing the first {n} data points (per rec_id) is: {len(df_signals_new)}')

    df_signals_new = df_signals_new.drop(columns=['time_step'])

    return df_signals_new


def remove_last_n_samples_of_signals(df_signals: pd.DataFrame, n: int = 3600) -> pd.DataFrame:
    """Remove the last n samples per rec_id because of transient effects of the filtering.

    When using filtered channels, note that the first and last 180 seconds of the signals should be ignored since
    these intervals contain transient effects of the filters: https://physionet.org/content/tpehgdb/1.0.1/#ref1

    Parameters
    ----------
    df_signals : pd.DataFrame
        Filtered signal data.
    n : int
        The number of last samples you want to remove from the filtered data.

    Returns
    -------
    df_signals_new : pd.DataFrame
        Dataframe that contains the signals with the last n samples removed (per rec_id).
    """
    print(f'The number of data points before removing the last {n} data points (per rec_id) is: {len(df_signals)}')

    df_signals_new = df_signals.loc[~df_signals
                                    .index
                                    .isin(df_signals
                                          .groupby([c.REC_ID_NAME])
                                          .tail(n)
                                          .index
                                          .values)].reset_index(drop=True).copy()

    print(f'The number of data points after removing the last {n} data points (per rec_id) is: {len(df_signals_new)}')

    return df_signals_new


def get_feature_names(column_transformer: Union[sklearn.pipeline.Pipeline,
                                                sklearn.compose._column_transformer.ColumnTransformer]) -> List[str]:
    """Get feature names from all transformers.

    Source: https://johaupt.github.io/blog/columnTransformer_feature_names.html

    Parameters
    -------
    column_transformer : Union[sklearn.pipeline.Pipeline,
    sklearn.compose._column_transformer.ColumnTransformer]

    Returns
    -------
    feature_names : List[str]
        Names of the features produced by transform.
    """

    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
            # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            # warnings.warn("Transformer %s (type %s) does not "
            #               "provide get_feature_names. "
            #               "Will return input column names if available"
            #               % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names_out()]

    # Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names


def calculate_peak_frequency(signal: np.array, fs: int) -> float:
    """Calculate the peak frequency of signal.

    The peak frequency represents the peak of the power distribution in the PSD and is the frequency that occurs
    most often in the power in the signal.

    Parameters
    ----------
    signal : np.array
        Signal over which you want to calculate the peak frequency.
    fs : int
        Sampling rate of signal.

    Returns
    -------
    peak_freq : float
        Peak frequency of the signal
    """
    # Calculate the discrete Fast Fourier Transform over the signal
    # FFT provides us spectrum density(i.e. frequency) of the time-domain signal
    fourier_transform = np.fft.rfft(signal)
    # PSD is defined as taking the square of the absolute value of FFT
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, (fs / 2), len(power_spectrum))
    # Peak frequency is the most occurring frequency
    peak_freq = frequency[np.argmax(power_spectrum)]

    return peak_freq


def calculate_median_frequency(signal: np.array, fs: int) -> float:
    """Calculate the median frequency of signal.

    The median frequency represents the midpoint of the power distribution in the PSD and is the frequency below
    and above which lies 50% of the total power in the signal.

    Parameters
    ----------
    signal : np.array
        Signal over which you want to calculate the median frequency.
    fs : int
        Sampling rate of signal.

    Returns
    -------
    median_freq : float
        Median frequency of the signal
    """
    # Calculate the discrete Fast Fourier Transform over the signal
    fourier_transform = np.fft.rfft(signal)
    # Calculate the discrete Fast Fourier Transform over the signal
    # FFT provides us spectrum density(i.e. frequency) of the time-domain signal
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, (fs / 2), len(power_spectrum))

    percent = 0.5
    cumsum = np.cumsum(power_spectrum)
    break_point = cumsum[-1] * percent
    median_freq = frequency[np.argmax(cumsum >= break_point) + 1]

    return median_freq


def calculate_feature_over_fixed_time_window(df_signals: pd.DataFrame, fixed_seq_length: int,
                                             column_list: List[str], feature_name: str, fs: int) -> pd.DataFrame:
    """Calculate a feature (sample entropy, peak or median frequency) over fixed_seq_length time windows over each
    column in column_list. In effect, you reduce the total sequence length to fixed_seq_length.

    Example: If the total sequence length of a rec_id is 27790 time steps and the fixed_seq_length is set at 300
    time steps, then the sample entropy/peak freq will be calculated over 300 consecutive subsequences. Each
    subsequence will consist of approximately the same number of time steps (and in this case the total number of
    time steps present in the subsequences sums up to 27790). Thus, the subsequences over which the samp en/peak freq
    is calculated will not have consisted of the exact same number of time steps.

    Parameters
    ----------
    df_signals : pd.DataFrame
        Original signal data of the features. Does not include the target. Must contain the column 'rec_id'.
    fixed_seq_length : int
        The number of sub sequences over which you want to calculate the samp en/peak frequency/median frequency.
    column_list : List[str]
        List with the names of the columns over which you want to calculate the feature_name (samp en/ peak freq).
    feature_name : str
        Name of the feature you want to calculate. Either 'sample_entropy', 'peak_frequency' or 'median_frequency'.
    fs : int
        Sampling rate in df_signals.

    Returns
    -------
    df_feature : pd.DataFrame
        Dataframe that contains the feature values for each subsequence for each rec_id. This dataframe
        will contain num_rec_ids*fixed_seq_length rows.
    """
    feature_options = ['sample_entropy', 'peak_frequency', 'median_frequency']
    assert feature_name in feature_options, f'{feature_name} is currently not implemented, choose ' \
                                            f'either of {feature_options}'
    num_rec_ids = df_signals[c.REC_ID_NAME].nunique()
    num_features = len(column_list)

    sequence_length_rec_ids = df_signals \
        .groupby(c.REC_ID_NAME, sort=False) \
        .agg({c.REC_ID_NAME: 'size'}) \
        .rename(columns={c.REC_ID_NAME: 'count'}) \
        .reset_index()

    # The results will be saved in this np.array and at the end converted to a dataframe
    x_arr_final = np.zeros((fixed_seq_length * num_rec_ids, num_features + 1))  # +1 is for the rec_id column
    start_index = 0

    # This code block contains several for loops in which we loop over:
    # - Each rec_id
    # - Each column
    # - Each subsequence
    # To calculate the sample entropy for each rec_id, for each feature and for each subsequence
    for i, rec_id in enumerate(sequence_length_rec_ids[c.REC_ID_NAME]):
        rec_id = int(rec_id)
        rec_id_feature = []

        for j, column in enumerate(column_list):
            # array_split splits an array into multiple sub-arrays of approx. the same length.
            # We use fixed_seq_length as the number of sub-arrays
            chunked_rec_id = np.array_split(df_signals.loc[df_signals[c.REC_ID_NAME] == rec_id, [column]],
                                            fixed_seq_length)
            feature = []

            for k in range(len(chunked_rec_id)):
                if feature_name == 'sample_entropy':
                    feature_chunk = ant.sample_entropy(chunked_rec_id[k].to_numpy().flatten())
                elif feature_name == 'peak_frequency':
                    feature_chunk = calculate_peak_frequency(chunked_rec_id[k].to_numpy().flatten(), fs=fs)
                elif feature_name == 'median_frequency':
                    feature_chunk = calculate_median_frequency(chunked_rec_id[k].to_numpy().flatten(), fs=fs)

                feature.append(feature_chunk)
            rec_id_feature.append(feature)
        # feature_arr_rec_id contains the feature_name for all features and each feature has its own column
        feature_arr_rec_id = np.column_stack(rec_id_feature)
        # Here we add the rec_id as a column
        feature_arr_rec_id = np.concatenate((np.array([rec_id] * fixed_seq_length).reshape(fixed_seq_length, -1),
                                             feature_arr_rec_id), axis=1)

        x_arr_final[start_index:(start_index + fixed_seq_length)] = feature_arr_rec_id
        start_index += fixed_seq_length

    df_feature = pd.DataFrame(x_arr_final, columns=[c.REC_ID_NAME] + column_list)
    df_feature[c.REC_ID_NAME] = df_feature[c.REC_ID_NAME].astype('int64')

    return df_feature


def feature_label_split(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the features from the label and return two separate dataframes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing both the features and the label.
    target_col : str
        Name of the target column.

    Returns
    -------
    [df_features, df_label] : Tuple[pd.DataFrame, pd.DataFrame]
    """
    df_label = df.loc[:, [target_col]]
    df_features = df.drop(columns=[target_col])

    return df_features, df_label


def train_val_test_split(df: pd.DataFrame, target_col: str,
                         test_ratio: np.float = 0.2, **kwargs: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                                             pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df in a train, val and test set. The val ration will be test_ratio / (1 - test_ratio). The split will
    be made with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing both the features and the label.
    target_col : str
        Name of the target column.
    test_ratio : np.float
        Ratio of df you want to allocate to the test set.
    **kwargs : Any
        Keyword arguments for the train_test_split function of sklearn.

    Returns
    -------
    [x_train, x_val, x_test, y_train, y_val, y_test] : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    val_ratio = test_ratio / (1 - test_ratio)
    df_features, df_label = feature_label_split(df, target_col)
    x_train, x_test, y_train, y_test = train_test_split(df_features, df_label, test_size=test_ratio,
                                                        stratify=df_label, **kwargs)
    # For stratification for the train and val split we now use y_train instead of df_label
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio, stratify=y_train, **kwargs)

    x_train = x_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return x_train, x_val, x_test, y_train, y_val, y_test


def preprocess_static_data(x_data_to_be_fitted: pd.DataFrame,
                           x_data_to_be_transformed: pd.DataFrame,
                           threshold_correlation: float = 0.85) -> [np.array, np.array, List[str], List[int]]:
    """Preprocess the static data.

    - Impute the missing values (with either the median or mean)
    - Standard scale data
    - Remove of a column pair if the correlation is > threshold_correlation

    Parameters
    ----------
    df_static_information : pd.DataFrame
        Dataframe containing the static data of each rec id.
    x_data_to_be_fitted : pd.DataFrame
        Dataframe that contains the train data. This data will be used to standard scale the data and will be applied
        on x_data_to_be_transformed.
    x_data_to_be_transformed : pd.DataFrame
        Dataframe that contains the data that needs to be transformed. Can be the same as x_data_to_be_fitted if it is
        the train data.
    threshold_correlation : float
        Threshold for which you want to remove one of the columns of a pair if the
        correlation is > threshold_correlation. Default is 85%.
    Returns
    -------
    X_arr_static, y_arr_static, selected_columns, rec_id_list_fit, rec_id_list_transform : Tuple[np.array, np.array,
    List[str], List[int], List[int]]
        X array with the preprocessed features and y array with the preprocessed labels.
        List with the feature names after preprocessing and the list of rec_ids present in
        the x fit array and x transform array.
    """
    # This df contains the static data
    # x_fit_static = df_static_information[df_static_information[c.REC_ID_NAME].isin(x_data_to_be_fitted[c.REC_ID_NAME].unique())] \
    #     .reset_index(drop=True).copy()
    # x_transform_static = df_static_information[df_static_information[c.REC_ID_NAME].isin(x_data_to_be_transformed[c.REC_ID_NAME].unique())] \
    #     .reset_index(drop=True).copy()

    # Keep the list of rec ids present as we want to know which data belongs to a certain id
    #rec_id_list_fit = x_data_to_be_fitted[c.REC_ID_NAME].unique()
    rec_id_list_transform = x_data_to_be_transformed[c.REC_ID_NAME].unique()

    # We need to this step because the X dataframe now contain the label column ('premature')
    x_fit_static, y_fit_static = feature_label_split(x_data_to_be_fitted, 'premature')
    x_transform_static, y_transform_static = feature_label_split(x_data_to_be_transformed, 'premature')

    # define column groups with same processing
    num_median_vars = ['parity', 'abortions']
    num_mean_vars = ['age', 'weight']
    num_remaining_vars = ['gestation_at_rec_time']

    # set up pipelines for each column group
    numeric_mean_pipe = Pipeline([('simple_imp_mean', SimpleImputer(strategy='mean')),
                                  ('standard_scaler', StandardScaler())])
    numeric_median_pipe = Pipeline([('simple_imp_median', SimpleImputer(strategy='median')),
                                    ('standard_scaler', StandardScaler())])
    numeric_remaining_pipe = Pipeline([('standard_scaler', StandardScaler())])

    # Set up columnTransformer
    # If using 'passthrough' for the remainder parameter, the columns when performing fit and transform must be in
    # the exact same order
    static_transformer = ColumnTransformer(
        transformers=[
            ('num_median_vars', numeric_median_pipe, num_median_vars),
            ('num_mean_vars', numeric_mean_pipe, num_mean_vars),
            ('num_remaining_vars', numeric_remaining_pipe, num_remaining_vars)],
        remainder='drop'
    )

    assert all(x == y for x, y in zip(x_fit_static.columns, x_transform_static)), "Columns in x_fit_static and " \
                                                                                  "x_transform_static must be in the" \
                                                                                  "exact same order!"

    lb = LabelEncoder()
    # Fit the transformer on the train data (x_fit_static) and then only transform on val and test data
    x_fitted_arr_static = fit_and_transform_data(x_fit_static, x_fit_static, static_transformer)
    x_transformed_arr_static = fit_and_transform_data(x_fit_static, x_transform_static, static_transformer)
    y_transformed_arr_static = fit_and_transform_data(y_fit_static.values.ravel(),
                                                      y_transform_static.values.ravel(), lb)

    # List of new feature names after transforming
    feature_list = get_feature_names(static_transformer)

    one_hot_enc_columns = [col for col in x_fit_static.columns if 'one_hot_encoder' in col]

    x_fitted_arr_static = pd.concat([pd.DataFrame(x_fitted_arr_static, columns=feature_list),
                                     x_fit_static[one_hot_enc_columns]], axis=1).reset_index(drop=True)

    x_transformed_arr_static = pd.concat([pd.DataFrame(x_transformed_arr_static, columns=feature_list),
                                          x_transform_static[one_hot_enc_columns]], axis=1).reset_index(drop=True)

    total_cols = feature_list + one_hot_enc_columns

    # We remove one if the correlated column pairs that have a correlation higher than threshold_correlation
    # from x_fitted_arr_static
    selected_columns = remove_correlated_columns(x_fitted_arr_static,
                                                 cols_to_check=x_fitted_arr_static.columns,
                                                 thresh=threshold_correlation)

    # Create a boolean series of which columns to include and which to exclude from x_fitted_arr_static
    boolean_series_selected_feat = pd.DataFrame(x_fitted_arr_static,
                                                columns=total_cols).columns.isin(selected_columns)

    # Only with the selected features from x_fitted_arr_static
    x_transformed_arr_static = x_transformed_arr_static.iloc[:, boolean_series_selected_feat]

    x_transformed_arr_static = x_transformed_arr_static.to_numpy()

    return x_transformed_arr_static, y_transformed_arr_static, selected_columns, rec_id_list_transform


def preprocess_signal_data(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train, y_test, features_to_use: List[str]):
    # Drop the rec_id column from the X dataframe
    column_drop_pipeline = Pipeline([("columnDropper", ColumnDropperTransformer([c.REC_ID_NAME]))])

    # apply the column drop pipeline to dataframe
    x_arr_test = fit_and_transform_data(x_train, x_test, column_drop_pipeline)
    x_arr_train = fit_and_transform_data(x_train, x_train, column_drop_pipeline)

    # ColumnTransformer needs a pd.DataFrame if columns are provided so we re-convert to a pd.DataFrame
    x_train = pd.DataFrame(data=x_arr_train, index=np.arange(len(x_arr_train)),
                           columns=[col for col in features_to_use if col != c.REC_ID_NAME])

    x_test = pd.DataFrame(data=x_arr_test,
                          index=np.arange(len(x_arr_test)),
                          columns=[col for col in features_to_use if col != c.REC_ID_NAME])

    # set up pipeline for the signal data
    numeric_standard_pipe = Pipeline([('standard_scaler', StandardScaler())])

    # Set up columnTransformer
    # If using 'passthrough' for the remainder parameter, the columns when performing fit and transform must be in
    # the exact same order. We use the static_transformer only for the signals data (present in columns_to_use)
    static_transformer = ColumnTransformer(transformers=[('num_signal_vars', numeric_standard_pipe, features_to_use)],
                                           remainder='passthrough')

    x_arr_test = fit_and_transform_data(x_train, x_test, static_transformer)
    x_arr_train = fit_and_transform_data(x_train, x_train, static_transformer)

    y_arr_test = fit_and_transform_data(y_train.values.ravel(),
                                        y_test.values.ravel(), LabelEncoder())

    y_arr_train = fit_and_transform_data(y_train.values.ravel(),
                                         y_train.values.ravel(), LabelEncoder())

    return x_arr_train, x_arr_test, y_arr_train, y_arr_test


def sort_in_descending_order_of_occurrence_count(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Sort df in descending order of occurrence count of 'column'.

    Parameters
    ----------
    df : pd.DataFrame
        Original signal data of the features.
    column : str
        Name of the column which you want to sort in order of occurrence count.

    Returns
    -------
    df : pd.DataFrame
        Sorted dataframe
    """
    # occurrence_count is the number of occurrence for each entry in 'column'
    df = df.assign(occurrence_count=df.groupby(column)
                   .rec_id\
                   .transform('count'))\
        .sort_values(['occurrence_count', column], ascending=[False, True])\
        .drop(columns=['occurrence_count'])\
        .reset_index(drop=True).copy()

    return df


def compute_non_padded_timesteps_per_batch_of_original_seq(df_features: pd.DataFrame,
                                                           fixed_seq_length: int,
                                                           sub_seq_length: int) -> pd.DataFrame:
    """Compute how many time steps of the original sequence are present in all the batches needed
    to process an entire padded sequence.

    We need to know the original length of the sequences before padding. This information is needed
    to use the pack_padded_sequence, as this function requires the batch to be sorted according to the lengths.

    Example:

    The fixed_seq_length variable tells us to which length all sequences will be padded to. The sub_seq_length
    variable tells us how many time steps are fed into the stateful LSTM model at the same time (i.e., per batch).
    If fixed_seq_length = 28800 and sub_seq_length = 200, then there are 28800 / 200 = 144 batches needed to
    process an entire padded sequence.

    We will calculate for each of the 144 batches how many time steps are filled up by the original (non-padded)
    sequence. For instance, if an original sequence is 27790 time steps, then 143 batches are filled up with
    200 original timesteps, and the last batch (batch 144) is filled up with 190 original time steps. The
    remaining 10 time steps in the last batch will later on be padded with zeros.

    The resulting dataframe for this specific sequence will look like this:

    |index|rec_id|batch_timesteps
       0      1       200
       1      1       200
       2      1       200
       3      1       200
            ....
      142     1       200
      143     1       190

    The order of the returned dataframe is therefore very important, as it is the exact order in which the
    stateful LSTM will process the batches.

    Parameters
    ----------
    df_features : pd.DataFrame
        Original signal data of the features. Does not include the target. Must contain the column 'rec_id'.
        This must be one of train/val/test set. (So not train/val/test combined).
    fixed_seq_length : int
        Universal length to which the original sequences are padded to.
    sub_seq_length : int
        Number of time steps the stateful LSTM model will process at the time.

    Returns
    -------
    batch_timesteps : pd.DataFrame
        Dataframe that contains for each batch for each rec id how many original time steps are fitted in that
        batch. The column batch_timesteps contains the number of original time steps for that batch.
        Note that the order of this dataframe is very important and must therefore not be shuffled.
    """
    sequence_length_rec_ids = df_features \
        .groupby(c.REC_ID_NAME, sort=False) \
        .agg({c.REC_ID_NAME: 'size'}) \
        .rename(columns={c.REC_ID_NAME: 'count'}) \
        .reset_index()

    # This list contains the original sequence length of each rec id
    sequence_length_list = list(sequence_length_rec_ids['count'])
    rec_ids = list(sequence_length_rec_ids[c.REC_ID_NAME])

    # The number of sub-sequences needed to make up an original sequence
    num_sub_sequences_fixed = fixed_seq_length / sub_seq_length

    rec_ids_list = np.repeat(rec_ids, num_sub_sequences_fixed)

    batches_all_rec_ids = []

    for i, seq_length in enumerate(sequence_length_list):

        # We calculate how many sub-sequences can be filled up entirely by the original sequence length
        num_sub_seq_rec_id = math.floor(seq_length / sub_seq_length)

        # If there are more sub-sequences that can be filled up than needed, then we know that the original
        # sequence will be truncated instead of padded. Therefore, each batch will be entirely filled up by the
        # original time steps
        if num_sub_seq_rec_id >= num_sub_sequences_fixed:
            total_batches = [sub_seq_length] * int(num_sub_sequences_fixed)

        # If not, then the original sequence will need to be padded, and we'll calculate how
        # many original time steps still fit in the last batch
        else:
            # These are all the batches that are completely filled up
            full_batches = [sub_seq_length] * math.floor(seq_length / sub_seq_length)

            # This is the remainder of the original time steps that fits in the last batch
            remainder = seq_length - math.floor(seq_length / sub_seq_length) * sub_seq_length
            full_batches.append(remainder)

            num_batches_to_be_filled = num_sub_sequences_fixed - len(full_batches)

            # The empty_batches means that there are only padded values in that batch (so no original time steps)
            # The value is therefore zero (no original time steps)
            empty_batches = [0] * int(num_batches_to_be_filled)

            total_batches = full_batches + empty_batches

            assert np.sum(total_batches) == seq_length, f"The calculation of the length of the batches for " \
                                                        f"rec id {rec_ids[i]} is incorrect"

        batches_all_rec_ids.append(total_batches)

    flattened_list = list(np.array(batches_all_rec_ids).flat)

    batch_length = pd.concat([pd.DataFrame(rec_ids_list, columns=[c.REC_ID_NAME]),
                              pd.DataFrame(flattened_list, columns=['batch_timesteps'])], axis=1)

    return batch_length


def custom_sort_seq_lengths(df_original_seq_lengths: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Custom sort df_original_seq_lengths such that we have the exact same order as when applying
    custom_sort_for_stateful_lstm(). See the documentation of custom_sort_for_stateful_lstm for more
    detailed explanation.

    We need to know the same order for the batches lengths, such that we can apply this for the
    pack_padded_sequence function.

    Parameters
    ----------
    df_original_seq_lengths : pd.DataFrame
        Dataframe that contains the batch lengths for each rec id. Result of
        compute_non_padded_timesteps_per_batch_of_original_seq()
    batch_size : int
        Number of rec ids present in one batch.

    Returns
    -------
    df_custom_sorted_lengths : pd.DataFrame
        Dataframe that contains for each batch for each rec id how many original time steps are fitted in that
        batch but now custom sorted.
    """
    num_rec_ids = df_original_seq_lengths[c.REC_ID_NAME].nunique()

    # The batch_size is the number of rec_ids for which you first want to sort the array correctly. After the
    # first batch_size of rec_ids is processed, we move onto the next batch_size rec_ids, up until we have finished
    # this process for all rec_ids. Example: if the num_rec_ids is 180 and the batch size is 60, rec_id_batches
    # will look like: [0, 60, 120], with these integers representing the first rec_id in the batch.
    rec_id_batches = list(range(0, num_rec_ids, batch_size))

    df_custom_sorted_lengths = pd.DataFrame()

    sorted_rec_ids = df_original_seq_lengths.groupby(c.REC_ID_NAME, sort=False)\
        .agg({c.REC_ID_NAME: 'size'})\
        .rename(columns={c.REC_ID_NAME: 'count'}).reset_index()[c.REC_ID_NAME]

    # Custom sort the batches per 60 rec ids. Meaning that we will first custom sort for
    # the first 60 rec ids, then the second 60 rec ids, etc.
    for rec_id_batch in rec_id_batches:
        tmp_rec_ids = sorted_rec_ids[rec_id_batch:(rec_id_batch + batch_size)]
        # tmp_batch contains the batch lengths for the tmp_rec_ids
        tmp_batch = df_original_seq_lengths[df_original_seq_lengths[c.REC_ID_NAME].isin(tmp_rec_ids)].copy()
        # Add the column 'occurrence', which contains the cumulative count of the occurrence of each
        # rec id. As each rec id has the same occurrence frequency, for each rec id the occurrence
        # goes from 0 until the number of batches needed to complete the entire sequence - 1.
        # So if there are 144 batches needed, then the occurrence goes from 0 till 143.
        tmp_batch.loc[:, 'occurrence'] = tmp_batch.groupby([c.REC_ID_NAME]).cumcount()
        # tmp_batch['occurrence'] = tmp_batch.groupby([c.REC_ID_NAME]).cumcount()
        tmp_batch = tmp_batch.rename_axis('sorted_index').sort_values(by=['occurrence', 'sorted_index'],
                                                                      ascending=[True, True])

        # In df_custom_sorted_lengths the final result when all rec ids are custom sorted will be saved
        df_custom_sorted_lengths = pd.concat([df_custom_sorted_lengths, tmp_batch], axis=0)

    # We do a check if the custom sort is processed correctly by checking for the first and last batch_size rec ids
    assert np.array_equal(sorted_rec_ids[-batch_size:].values,
                          df_custom_sorted_lengths[-batch_size:][c.REC_ID_NAME].values), \
        f"Custom sort is incorrect for the last {batch_size} rec ids."

    assert np.array_equal(sorted_rec_ids[:batch_size].values,
                          df_custom_sorted_lengths[:batch_size][c.REC_ID_NAME].values), \
        f"Custom sort is incorrect for the first {batch_size} rec ids."

    df_custom_sorted_lengths = df_custom_sorted_lengths.reset_index()

    return df_custom_sorted_lengths


def pad_sequences_to_fixed_length(df_features: pd.DataFrame, x_arr: np.ndarray,
                                  y_arr: np.ndarray, num_features: int = 12,
                                  fixed_seq_length: int = 28800) -> Tuple[np.ndarray, np.ndarray]:
    """Pad the total sequence length of each rec_id to fixed_seq_length.

    If the total sequence length of a rec_id is smaller than fixed_seq_length then the sequence is
    padded with zeros up until fixed_seq_length. If the total sequence length is larger, then the
    sequence gets truncated to fixed_seq_length.

    Parameters
    ----------
    df_features : pd.DataFrame
        Original signal data of the features. Does not include the target. Must contain the column 'rec_id'.
    x_arr : np.ndarray
        Array with the scaled data of the features.
    y_arr : np.ndarray
        Array with the labels (not scaled).
    num_features : int
        The number of features present in the data.
    fixed_seq_length : int
        Universal length you want to pad/truncate the sequences to.

    Returns
    -------
    X_arr_final : np.ndarray
        Array that contains the feature values for each rec_id but now padded/truncated to fixed_seq_length for
        each rec_id.
    y_arr_final : np.ndarray
        Array that contains the labels for each rec_id but now padded/truncated to fixed_seq_length for each rec_id.
    """
    sequence_length_rec_ids = df_features\
        .groupby(c.REC_ID_NAME, sort=False)\
        .agg({c.REC_ID_NAME: 'size'})\
        .rename(columns={c.REC_ID_NAME: 'count'})\
        .reset_index().copy()

    num_rec_ids = df_features[c.REC_ID_NAME].nunique()
    start_index = 0

    x_arr_final = np.zeros((fixed_seq_length * num_rec_ids, num_features))
    y_arr_final = np.zeros((fixed_seq_length * num_rec_ids), dtype=np.int64)

    for i, rec_id in enumerate(sequence_length_rec_ids[c.REC_ID_NAME]):
        rec_id = int(rec_id)
        seq_length_rec_id = sequence_length_rec_ids[sequence_length_rec_ids[c.REC_ID_NAME] == rec_id]['count'].values[0]

        if seq_length_rec_id < fixed_seq_length:
            end_index = start_index + seq_length_rec_id
            x_arr_rec_id = x_arr[start_index:end_index]
            y_arr_rec_id = y_arr[start_index:end_index]

            # We pad the X values with zeros up until fixed_seq_length
            x_arrs_to_add = np.repeat(np.array([[0.0] * num_features]), (fixed_seq_length - seq_length_rec_id), axis=0)
            x_arr_rec_id = np.concatenate((x_arr_rec_id, x_arrs_to_add), axis=0)

            # For the y values it holds that we take the label belonging to the rec_id
            # This label is the same throughout the entire sequence, so we can take any value between
            # start_index and end_index (as this value is constant)
            y_arrs_to_add = [y_arr[start_index]] * (fixed_seq_length - seq_length_rec_id)
            y_arr_rec_id = np.concatenate((y_arr_rec_id, y_arrs_to_add), axis=0)

        else:
            # If the total seq length of a rec_id is longer than fixed_seq_length, we drop
            # all the values between fixed_seq_length and seq_length_rec_id
            x_arr_rec_id = x_arr[start_index:(start_index + fixed_seq_length)]
            y_arr_rec_id = y_arr[start_index:(start_index + fixed_seq_length)]

        assert len(x_arr_rec_id) == fixed_seq_length, f'X values of {c.REC_ID_NAME} {rec_id} ' \
                                                      f'are not properly padded/truncated'
        assert len(y_arr_rec_id) == fixed_seq_length, f'Y values of {c.REC_ID_NAME} {rec_id} ' \
                                                      f'are not properly padded/truncated'

        x_arr_final[i * fixed_seq_length:((i * fixed_seq_length) + fixed_seq_length)] = x_arr_rec_id

        y_arr_final[i * fixed_seq_length:((i * fixed_seq_length) + fixed_seq_length)] = y_arr_rec_id

        start_index += seq_length_rec_id

    return x_arr_final, y_arr_final


def pad_sequences_to_fixed_length_df(df_features: pd.DataFrame, num_features: int = 12,
                                     fixed_seq_length: int = 28800) -> pd.DataFrame:
    """Pad the total sequence length of each rec_id to fixed_seq_length.

    If the total sequence length of a rec_id is smaller than fixed_seq_length then the sequence is
    padded with zeros up until fixed_seq_length. If the total sequence length is larger, then the
    sequence gets truncated to fixed_seq_length.

    This function is created to check the output of the custom_sort_for_lstm_stateful() function.

    Parameters
    ----------
    df_features : pd.DataFrame
        Original signal data of the features. Does not include the target. Must contain the column 'rec_id'.
    num_features : int
        The number of features present in the data.
    fixed_seq_length : int
        Universal length you want to pad/truncate the sequences to.

    Returns
    -------
    df_final : pd.DataFrame
        Dataframe that contains the feature values for each rec_id but now padded/truncated to fixed_seq_length for
        each rec_id.
    """
    sequence_length_rec_ids = df_features.groupby(c.REC_ID_NAME, sort=False)\
        .agg({c.REC_ID_NAME: 'size'})\
        .rename(columns={c.REC_ID_NAME: 'count'})\
        .reset_index().copy()

    start_index = 0
    df_final = pd.DataFrame()

    for i, rec_id in enumerate(sequence_length_rec_ids[c.REC_ID_NAME]):
        rec_id = int(rec_id)
        seq_length_rec_id = sequence_length_rec_ids[sequence_length_rec_ids[c.REC_ID_NAME] == rec_id]['count'].values[0]

        if seq_length_rec_id < fixed_seq_length:
            end_index = start_index + seq_length_rec_id
            x_train_rec_id = df_features[start_index:end_index]

            rows_to_add = ([[rec_id] + [0.] * num_features]) * (fixed_seq_length - seq_length_rec_id)
            x_train_rec_id = pd.concat([x_train_rec_id, pd.DataFrame(rows_to_add, columns=df_features.columns)])

        else:
            # If the total seq length of a rec_id is longer than fixed_seq_length, we drop
            # all the values between fixed_seq_length and seq_length_rec_id
            x_train_rec_id = df_features[start_index:(start_index + fixed_seq_length)]

        assert len(x_train_rec_id) == fixed_seq_length, f'X values of {c.REC_ID_NAME} {rec_id} are not ' \
                                                        f'properly padded/truncated'

        df_final = pd.concat([df_final, x_train_rec_id], ignore_index=True)

        start_index += seq_length_rec_id

    return df_final


def custom_sort_for_stateful_lstm(df_features: pd.DataFrame, x_arr_padded: np.ndarray,
                                  y_arr_padded: np.ndarray, fixed_seq_length: int = 28800,
                                  sub_seq_length: int = 200, batch_size: int = 60,
                                  num_features: int = 12) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Custom sort x_arr_padded and y_arr_padded such that we can correctly batchify the data using the
    DataLoader class.

    In order to create a stateful LSTM model, we must put the data in the correct format. Stateful means
    that the last state for each sample at index i in a batch will be used as initial state for the sample of
    index i in the following batch.

    This means that the batches we want to feed to the model must have this form:

    batch1 = [sample10, sample20, sample30, sample40]
    batch2 = [sample11, sample21, sample31, sample41]

    and not this form:

    batch1 = [sample10, sample20, sample30, sample40]
    batch2 = [sample21, sample41, sample11, sample31]

    This implies 21 causally follows 10 - and will wreck training.
    (For discussion see: https://github.com/lmnt-com/haste/issues/8)

    Also important to be aware of: samples in a batch are processed independently.
    (https://discuss.pytorch.org/t/handling-the-hidden-state-with-minibatches-in-a-rnn-for-language-modelling/44261/2)

    To translate this to our case:

    Let's say we want to process 60 rec_ids in parallel (so in the same batch), and also split up the
    entire sequence (28800 time steps) of these 60 rec ids in multiple sub-sequences of 200 time steps. We want to
    apply a stateful LSTM model, so we want to pass the states onto the next batch. If we have 180 rec ids in total,
    then we must first process the entire sequence of the first 60 rec_ids and then via a resettable method reset the
    states and only then move on to the next 60 rec ids.

    At this moment, x_arr_padded and y_arr_padded are ordered in ascending time steps for each rec_id separately.
    So: [t0,1; t1,1; t2,1; ..., t28800,1; t0,2; t1,2; t2,2; ..., t28800,2; ... t0,180; t1,180; ..., t28800,180],
    with tn, m: n the time step and m the rec id.

    In this function, we sort the x_arr_padded and y_arr_padded in such way that we take a batch of rec_ids
    (60 in our case) and take the first sub_seq_length time steps for the first rec id, then the first sub_seq_length
    time steps for the second rec id, ..., the last sub_seq_length time steps of the last rec_id (of the first batch
    of 60 rec ids). After that, we move on to the next batch of 60 rec ids and we take the first sub_seq_length
    time steps of rec id 61, etc.

    As a result, we will have an X array that looks like this:

    X = [[t0,1], [t1,1], [t2,1], [t3,1], ..., [t199,1], [t0,2], [t1,2], ..., [t199,2], ..., [t0,60], ..., [t199,60],
    [t200,1], ..., [t399,1], ..., [t200,60] ..., [t399,60], ..., [t28800,1], ..., [t28800,60], ..., [t0,61] ...,
    [t199,61], ..., [t0,121] ..., [t28800,180]]

    With tn,m: n the time step, m the rec id.

    After we have sorted the X and y array in this way we can correctly batchify our data (using the DataLoader class
    of PyTorch) for a stateful model.

    Parameters
    ----------
    df_features : pd.DataFrame
        Original signal data of the features. Does not include the target. Must contain the column 'rec_id'.
    x_arr_padded : np.ndarray
        Array with the scaled and padded data of the features.
    y_arr_padded : np.ndarray
        Array with the padded labels (not scaled).
    fixed_seq_length : int
        Universal length to which the sequence of each rec_id is padded/truncated.
    sub_seq_length : int
        Number of time steps you want to feed to the stateful LSTM model for each batch for each rec id. Must be a
        multiple of fixed_seq_length.
    batch_size : int
        The number of rec_ids for which you first want to complete the entire sequence.
    num_features : int
        The number of features present in the data.

    Returns
    -------
    X_arr_final : np.ndarray
        Array that contains the feature values for each rec_id but now custom sorted.
    y_arr_final : np.ndarray
        Array that contains the labels for each rec_id but now custom sorted.
    batch_sizes : List[int]
        List with the number of rec_ids present per batch. For instance, if the number of
        rec ids in your df_features is 180 and the batch_size is 50, then batch_sizes
        will be [50, 50, 50, 30].
    """
    num_rec_ids = df_features[c.REC_ID_NAME].nunique()

    # The custom sorted array
    x_sorted = np.zeros((fixed_seq_length * num_rec_ids, num_features))
    y_sorted = np.zeros((fixed_seq_length * num_rec_ids), dtype=np.int64)

    # The batch_size is the number of rec_ids for which you first want to sort the array correctly. After the
    # first batch_size of rec_ids is processed, we move onto the next batch_size rec_ids, up until we have finished
    # this process for all rec_ids. Example: if the num_rec_ids is 180 and the batch size is 60, rec_id_batches
    # will look like: [0, 60, 120], with these integers representing the first rec_id in the batch.
    rec_id_batches = list(range(0, num_rec_ids, batch_size))

    # time_step_blocks represents the number of sub_seq_length you need to fill the entire fixed_seq_length.
    # Example: If the fixed_seq_length is 28800 and the sub_seq_length is 200, you need 144 blocks of 200 to process
    # the entire fixed_seq_length
    time_step_blocks = math.ceil(fixed_seq_length / sub_seq_length)
    new_indexes_order = []

    num_processed_rec_ids = 0
    num_remaining_rec_ids = df_features[c.REC_ID_NAME].nunique()
    batch_sizes = []

    for rec_id_batch in rec_id_batches:
        # Check if there are enough rec_ids left to fill a full batch_size, if not then the batch_size will
        # be adjusted (we name this the 'true_batch_size')
        if num_remaining_rec_ids >= batch_size:
            true_batch_size = batch_size
        else:
            true_batch_size = num_remaining_rec_ids

        for block in range(0, time_step_blocks):
            for rec_id in range(rec_id_batch, rec_id_batch + true_batch_size):
                # sorted_indexes contains the indexes in the correct order for the batch_size rec_ids for the block time
                # steps. We will use sorted_indexes to extract the values in the correct order from X_arr_padded and
                # y_arr_padded
                sorted_indexes = [x for x in range((rec_id * fixed_seq_length) + block * sub_seq_length,
                                                   (rec_id * fixed_seq_length) + (block + 1) * sub_seq_length)]
                new_indexes_order.append(sorted_indexes)

        num_processed_rec_ids += true_batch_size
        num_remaining_rec_ids = num_rec_ids - num_processed_rec_ids
        batch_sizes.append(true_batch_size)

    assert np.sum(batch_sizes) == num_rec_ids, f'The total number of processed rec_ids is {np.sum(batch_sizes)}, ' \
                                               f'but should be {num_rec_ids}!'
    flattened_list = list(np.array(new_indexes_order).flat)

    x_sorted[0:(num_rec_ids * fixed_seq_length)] = x_arr_padded[flattened_list]
    y_sorted[0:(num_rec_ids * fixed_seq_length)] = y_arr_padded[flattened_list]

    return x_sorted, y_sorted, batch_sizes


class StaticDataset(Dataset):
    """Create Datasets from the data, where each index corresponds to a single item from the data."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class NonOverlappingSequencesDataset(Dataset):
    """Create a custom Dataset with non-overlapping sequences.

    !!This function can only be used if the dataset has been custom sorted. Meaning that the input dataset must be
    sorted in such way that a block of sub_seq_length time steps always belongs to a distinct rec_id!!

    Example: A dataset in which the total sequence consists of 6000 time steps. We want to
    split this total sequence of 6000 time steps up into distinct sub-sequences of 200 time steps each.
    The sub-sequences must have no overlap. Assume we start at start_point t=0, the resulting sub-sequences
    will then be:

    Time steps 0-199
    Time steps 200-399
    Time steps 400-599
    .
    .
    .
    Time steps 5800-5999.

    In this example the time steps all belong to the same rec_id, but you can also have an example in which the
    time step blocks belong to different rec ids (as long as within a time step block all the time steps belong to a
    distinct rec id).

    After creating this custom dataset, you can forward this to the DataLoader class and define a batch size.

    Parameters
    ----------
    data : TensorDataset
        Data (both the features and the target(s)) in TensorDataset format.
    sub_seq_length : int
        The number of time steps you want to include in each sub-sequence.
    num_features : int
        The number of features present in data.
    num_targets : int
        The number of target(s) present in data. Default is 1.
    start_point : int
        The time step at which you want to start dividing up the sequences in data into length sub_seq_length.
        Default is t=0.

    Returns
    -------
    data : List[torch.Tensor]
        List containing all the tensors of the features with length of sub_seq_length.
        Shape: torch.Size([sub_seq_length, num_features])
    target : List[torch.Tensor]
        List containing all the tensors of the target(s) with length of sub_seq_length.
        Shape: torch.Size([sub_seq_length, num_targets])
    """
    def __init__(self, data, sub_seq_length=200, num_features=12, num_targets=1, start_point=0):
        self.data = data
        self.sub_seq_length = sub_seq_length
        self.full_seq_length = len(self.data)
        self.num_features = num_features
        self.num_targets = num_targets
        self.start_point = start_point

    def __len__(self):
        # The __len__ function returns the number of samples in our dataset. As we want to divide the
        # total sequence into distinct sequence of length sub_seq_length, we end up with len(data) / sub_seq_length
        # samples in total. If the len(data) is not a multiple of sub_seq_length, then the last __len__ will contain
        # less samples than sub_seq_length.
        # If one wants to start from a different start_point than t=0, this is possible.
        # If we do not adjust the __len__ function, the dataset goes into infinite loop when used as iterator.
        return math.ceil((len(self.data) - self.start_point) / self.sub_seq_length)

    def __getitem__(self, index):
        # We use __get_item__ to create a data sample of the shape [self.sub_seq_len, self.num_features]
        # In order to get non-overlapping sequences, we have to multiply the index by the sub_seq_length
        index = index * self.sub_seq_length
        # We initialize the data and target variable with zeros and these variables will then be filled with
        # the actual data (self.data)
        data = torch.zeros(self.sub_seq_length, self.num_features)
        target = self.data[index + self.start_point][1]
        # The length of the data sample that will be returned is of size sub_seq_length
        for i in range(0, self.sub_seq_length):
            data[i] = self.data[index + i + self.start_point][0]

        return data, target


class CombinedDataset(Dataset):
    """Combine sequential and static data for each rec id into one Dataset"""
    def __init__(self, seq_data, static_data):
        self.seq_data = seq_data
        self.static_data = static_data

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, index):
        combined_data = self.seq_data[index][0], self.static_data[index][0]
        assert self.seq_data[index][1] == self.static_data[index][1], \
            f"Target of index {index} is not the same for both datasets!"
        # The target is present in both the seq and static data (and should be the same in both datasets obviously),
        # we need the target only once so, we take the target from the seq data (which is the same as the static data)
        target = self.seq_data[index][1]
        return combined_data, target


class ColumnDropperTransformer:
    def __init__(self, columns):
        self.columns = columns

    def transform(self, x, y=None):
        return np.array(x.drop(self.columns, axis=1))

    def fit(self, x, y=None):
        return self


def custom_calculate_feature_over_fixed_time_window(trial, params: Dict, df_signals_new: pd.DataFrame,
                                                    df_clinical_information: pd.DataFrame,
                                                    x_data_to_be_fitted: pd.DataFrame,
                                                    x_data_to_be_transformed: pd.DataFrame,
                                                    columns_to_use: List[str], feature_name: str,
                                                    fs: int, shuffle: bool = False) -> DataLoader:
    """Dynamically create a custom DataLoader with different hyperparameter values.

    The feature_name (samp en/peak freq/median freq) is calculated over reduced_seq_length time windows.

    Parameters
    ----------
    trial :
    params : Dict
        Dictionary containing the hyperparameters and its values of the current trial run.
    df_signals_new : pd.DataFrame
        DataFrame that contains the signal data after applying remove_first_n_samples_of_signals and
        remove_last_n_samples_of_signals.
    df_clinical_information : pd.DataFrame
        Dataframe that contains the label 'premature'.
    x_data_to_be_fitted : pd.DataFrame
        Dataframe that contains the data you want to use for data preprocessing such as standard scaling.
        In general, this should be x_train.
    x_data_to_be_transformed : pd.DataFrame
        Dataframe that contains the data you want to transform by using the Pipeline that is fitted on
        x_data_to_be_fitted. Can also be the same as x_data_to_be_fitted.
    columns_to_use : List[columns]
        Names of the features you want to use for modeling.
    feature_name : str
        Name of the feature you want to calculate. Either 'sample_entropy', 'peak_frequency' or 'median_frequency'.
    fs : int
        Sampling frequency of df_signals_new.
    shuffle : Boolean
        Whether to shuffle the DataLoader. Default is False. Shuffling is recommended during training.

    Returns
    -------
    custom_loader_feature : DataLoader
        DataLoader with batch_size samples and reduced sequences by calculating a feature (samp en/ peak/ median freq)
        for each rec_id.
    """
    df_feature = calculate_feature_over_fixed_time_window(df_signals_new,
                                                          fixed_seq_length=params['reduced_seq_length'],
                                                          column_list=columns_to_use,
                                                          feature_name=feature_name,
                                                          fs=fs)

    df_feature = df_feature.merge(df_clinical_information[[c.REC_ID_NAME, 'premature']], how='left', on=c.REC_ID_NAME)

    if feature_name == 'sample_entropy':

        df_feature = replace_inf_with_zero(df_feature)
        df_feature = replace_na_with_zero(df_feature)

    # This df contain the sample entropy data for the rec ids in x_data_to_be_transformed
    x_feature_transform = df_feature[df_feature[c.REC_ID_NAME].isin(x_data_to_be_transformed[c.REC_ID_NAME].unique())].reset_index(drop=True).copy()
    x_feature_fit = df_feature[df_feature[c.REC_ID_NAME].isin(x_data_to_be_fitted[c.REC_ID_NAME].unique())].reset_index(drop=True).copy()

    # We need to re-do this step because the X dataframes now contain the label column ('premature')
    x_feature_transform, y_feature_transform = feature_label_split(x_feature_transform, 'premature')
    x_feature_fit, y_feature_fit = feature_label_split(x_feature_fit, 'premature')

    # Drop the rec_id column from the X dataframe
    column_drop_pipeline = Pipeline([("columnDropper", ColumnDropperTransformer([c.REC_ID_NAME]))])

    # apply the pipeline to dataframe
    x_arr_feature_transformed = fit_and_transform_data(x_feature_fit, x_feature_transform, column_drop_pipeline)
    x_arr_feature_fitted = fit_and_transform_data(x_feature_fit, x_feature_fit, column_drop_pipeline)

    standard_scalar_pipe = Pipeline([('standard_scaler', StandardScaler())])
    x_arr_feature_transformed = fit_and_transform_data(x_arr_feature_fitted, x_arr_feature_transformed,
                                                       standard_scalar_pipe)

    y_arr_feature_transformed = fit_and_transform_data(y_feature_fit.values.ravel(),
                                                       y_feature_transform.values.ravel(), LabelEncoder())

    # Create tensors of the X and y data. BCEWithLogitsLoss requires its target to be a float tensor, not long.
    features = torch.from_numpy(x_arr_feature_transformed)
    targets = torch.from_numpy(y_arr_feature_transformed).float().view(-1, 1)

    # We put the features and targets in TensorDataset format
    dataset_feature = TensorDataset(features, targets)

    # Create custom Datasets with non-overlapping sequences of length reduced_seq_length
    custom_non_overlap_dataset_feature = NonOverlappingSequencesDataset(dataset_feature,
                                                                        params['reduced_seq_length'],
                                                                        len(columns_to_use),
                                                                        num_targets=1, start_point=0)

    # Please note that the batch_size must be <= len(data) / sub_seq_length. Otherwise, there are too few
    # datapoints to fit in the number of batches.
    # We shuffle the samples in the train_loader
    custom_loader_feature = DataLoader(custom_non_overlap_dataset_feature,
                                       batch_size=params['batch_size'],
                                       shuffle=shuffle, drop_last=False)

    return custom_loader_feature


def basic_preprocessing_signal_data(df_signals, df_clinical_information, reduced_seq_length, features_to_use, feature_name, fs):
    # This dataframe contains calculated features (e.g., peak frequency) over the original signal data
    df_features = calculate_feature_over_fixed_time_window(df_signals,
                                                           fixed_seq_length=reduced_seq_length,
                                                           column_list=features_to_use,
                                                           feature_name=feature_name,
                                                           fs=fs)

    if feature_name == 'sample_entropy':
        df_features = replace_inf_with_zero(df_features)
        df_features = replace_na_with_zero(df_features)

    # Re-add the label 'premature' to df_feature, which will now have as many time steps as the calculated features
    df_features = df_features.merge(df_clinical_information[[c.REC_ID_NAME, 'premature']], how='left', on=c.REC_ID_NAME)

    # Assign 0 to non-premature cases and assign 1 to premature cases
    lb = LabelEncoder()
    df_features, df_label = feature_label_split(df_features, 'premature')
    df_label['premature'] = lb.fit_transform(df_label['premature'])

    return df_features, df_label



def basic_preprocessing_static_data(data_path: str, settings_path: str, df_clinical_information: pd.DataFrame):
    lb = LabelEncoder()
    df_demographics = build_demographics_dataframe(data_path, settings_path)
    df_static_information = df_demographics.merge(df_clinical_information, how='left', on=c.REC_ID_NAME)
    df_static_information['premature'] = lb.fit_transform(df_static_information['premature'])

    # define column groups with same processing
    cat_vars = ['hypertension', 'diabetes', 'placental_position', 'bleeding_first_trimester',
                'bleeding_second_trimester', 'funneling', 'smoker']

    # set up pipelines for each column group
    categorical_pipe = Pipeline([('one_hot_encoder', OneHotEncoder(handle_unknown='error'))])

    # Set up columnTransformer
    # If using 'passthrough' for the remainder parameter, the columns when performing fit and transform must be in
    # the exact same order
    static_transformer = ColumnTransformer(transformers=[('cats', categorical_pipe, cat_vars)],
                                           remainder='passthrough')

    df_static_information = fit_and_transform_data(df_static_information, df_static_information, static_transformer)

    # List of new feature names after transforming
    feature_list = get_feature_names(static_transformer)

    # Manual remapping of column names back to their original name after using static_transformer for columns that
    # have not been transformed.
    rename_dict = {'x0': 'rec_id', 'x1': 'age', 'x2': 'parity', 'x3': 'abortions', 'x4': 'weight', 'x12': 'gestation',
                   'x13': 'gestation_at_rec_time', 'x14': 'group', 'x15': 'premature', 'x16': 'early'}

    df_static_information = pd.DataFrame(df_static_information, columns=feature_list)
    df_static_information = pd.DataFrame(df_static_information).rename(columns=rename_dict)

    # Cast one hot encoder columns to float64
    one_hot_enc_columns = [col for col in df_static_information.columns if 'one_hot_encoder' in col]
    df_static_information[one_hot_enc_columns] = df_static_information[one_hot_enc_columns].apply(pd.to_numeric)

    return df_static_information


def add_static_data_to_signal_data(X_train_static: pd.DataFrame, X_test_static: pd.DataFrame,
                                   X_train_signal: pd.DataFrame, X_test_signal: pd.DataFrame,
                                   X_train_signal_processed: np.array, X_test_signal_processed: np.array,
                                   rec_ids_x_train_signal: List[int], rec_ids_x_test_signal: List[int],
                                   features_to_use: List[str], threshold_correlation: float) -> Tuple[pd.DataFrame,
                                                                                                      pd.DataFrame,
                                                                                                      np.array,
                                                                                                      np.array,
                                                                                                      List[str]]:

    X_train_combined = X_train_signal.merge(X_train_static, on=c.REC_ID_NAME).reset_index(drop=True)
    X_test_combined = X_test_signal.merge(X_test_static, on=c.REC_ID_NAME).reset_index(drop=True)

    assert set(X_train_signal[c.REC_ID_NAME]) == set(X_train_static[c.REC_ID_NAME]), \
        "Rec ids in inner fold in train signals and train static data are not the same!"

    assert set(X_test_signal[c.REC_ID_NAME]) == set(X_test_static[c.REC_ID_NAME]), \
        "Rec ids in inner fold test signals and test static data are not the same!"

    # Count the number of static features within this specific training fold
    # One of a pair of highly correlated features (>85%) will be removed and this can be
    # different within each fold

    x_arr_static_train, y_arr_static_train, selected_columns_train_static, rec_id_list_static_train = preprocess_static_data(
        X_train_static,
        X_train_static,
        threshold_correlation=threshold_correlation)

    num_static_features = len(selected_columns_train_static)
    print(f'Num static features: {num_static_features}')

    x_arr_static_test, y_arr_static_test, selected_columns_test_static, rec_id_list_static_test = preprocess_static_data(
        X_train_static,
        X_test_static,
        threshold_correlation=threshold_correlation)

    assert set(selected_columns_train_static) == set(selected_columns_test_static), "Columns used for" \
                                                                                    "preprocessing static data" \
                                                                                    "is not equal for train" \
                                                                                    "and test!"
    # Safety check if the data consequently processed for the correct order of rec ids
    assert set(rec_id_list_static_test) == set(X_test_static[c.REC_ID_NAME].unique())
    assert set(rec_id_list_static_train) == set(X_train_static[c.REC_ID_NAME].unique())

    # Re-add column 'rec_id'
    df_static_train = pd.concat([pd.DataFrame(rec_id_list_static_train, columns=[c.REC_ID_NAME]),
                                 pd.DataFrame(x_arr_static_train, columns=selected_columns_train_static)],
                                axis=1)
    df_static_test = pd.concat([pd.DataFrame(rec_id_list_static_test, columns=[c.REC_ID_NAME]),
                                pd.DataFrame(x_arr_static_test, columns=selected_columns_test_static)], axis=1)

    X_train_signal_processed = pd.concat([pd.DataFrame(rec_ids_x_train_signal, columns=[c.REC_ID_NAME]),
                                          pd.DataFrame(X_train_signal_processed,
                                                       columns=features_to_use)], axis=1)

    X_test_signal_processed = pd.concat([pd.DataFrame(rec_ids_x_test_signal, columns=[c.REC_ID_NAME]),
                                         pd.DataFrame(X_test_signal_processed,
                                                      columns=features_to_use)], axis=1)

    X_train_combined_processed = X_train_signal_processed.merge(df_static_train,
                                                                on=c.REC_ID_NAME).reset_index(drop=True)
    X_test_combined_processed = X_test_signal_processed.merge(df_static_test,
                                                              on=c.REC_ID_NAME).reset_index(drop=True)

    # Drop the rec_id column from the X dataframe
    column_drop_pipeline = Pipeline([("columnDropper", ColumnDropperTransformer([c.REC_ID_NAME]))])
    X_train_combined_processed = fit_and_transform_data(X_train_combined_processed, X_train_combined_processed,
                                                        column_drop_pipeline)
    X_test_combined_processed = fit_and_transform_data(X_train_combined_processed, X_test_combined_processed,
                                                       column_drop_pipeline)

    return X_train_combined, X_test_combined, X_train_combined_processed, X_test_combined_processed, \
           selected_columns_train_static


def generate_dataloader(x_feature, x_preprocessed, y_preprocessed, features_to_use, features_to_use_static, rec_ids,
                        reduced_seq_length, sub_seq_length, num_sub_sequences, batch_size, test_phase):

    rec_ids_list = np.repeat(rec_ids, num_sub_sequences)

    batch_lengths = np.repeat([sub_seq_length], len(rec_ids) * num_sub_sequences)

    batch_length = pd.concat([pd.DataFrame(rec_ids_list, columns=[c.REC_ID_NAME]),
                              pd.DataFrame(batch_lengths, columns=['batch_timesteps'])], axis=1)

    x_arr_feature_transformed, y_arr_feature_transformed, batch_sizes = \
        custom_sort_for_stateful_lstm(x_feature, x_preprocessed, y_preprocessed,
                                      fixed_seq_length=reduced_seq_length,
                                      sub_seq_length=sub_seq_length,
                                      batch_size=batch_size,
                                      num_features=len(features_to_use) + len(features_to_use_static))

    # If the batch sizes are not equal across all batches we create two separate dataloaders
    if not batch_sizes.count(batch_sizes[0]) == len(batch_sizes):
        custom_loader_orig_part1, custom_loader_orig_part2, _, _ = \
            create_dataloader_if_batch_sizes_unequal(x_arr_feature_transformed, y_arr_feature_transformed,
                                                     batch_size, batch_sizes, num_sub_sequences,
                                                     batch_length,
                                                     features_to_use=features_to_use + features_to_use_static,
                                                     fixed_seq_length=reduced_seq_length,
                                                     sub_seq_length=sub_seq_length)

        if test_phase:
            return [custom_loader_orig_part1, custom_loader_orig_part2], rec_ids_list

        elif not test_phase:
            return [custom_loader_orig_part1, custom_loader_orig_part2]

    if batch_sizes.count(batch_sizes[0]) == len(batch_sizes):
        custom_loader_orig, _ = create_dataloader_if_batch_sizes_equal(x_arr_feature_transformed,
                                                                       y_arr_feature_transformed,
                                                                       batch_size,
                                                                       batch_length,
                                                                       features_to_use=features_to_use + features_to_use_static,
                                                                       sub_seq_length=sub_seq_length)

        if test_phase:
            return [custom_loader_orig], rec_ids_list
        elif not test_phase:
            return [custom_loader_orig]


def custom_calculate_feature_over_splited_fixed_time_window(trial, params, df_signals_new: pd.DataFrame,
                                                            df_clinical_information: pd.DataFrame,
                                                            df_static_information: pd.DataFrame,
                                                            x_data_to_be_fitted: pd.DataFrame,
                                                            x_data_to_be_transformed: pd.DataFrame,
                                                            df_static_train: pd.DataFrame,
                                                            df_static_test: pd.DataFrame,
                                                            selected_columns_static_data: List,
                                                            columns_to_use: List[str], feature_name: str,
                                                            reduced_seq_length, sub_seq_length,
                                                            fs: int, shuffle: bool = False,
                                                            add_static_data: bool = False, test_phase: bool = False):
    """Dynamically create a custom DataLoader with different hyperparameter values.

    The feature_name (samp en/peak freq/median freq) is calculated over reduced_seq_length (present in dict params)
    time windows.

    Parameters
    ----------
    trial :
    params : Dict
        Dictionary containing the hyperparameters and its values of the current trial run.
    df_signals_new : pd.DataFrame
        DataFrame that contains the signal data after applying remove_first_n_samples_of_signals and
        remove_last_n_samples_of_signals.
    df_clinical_information : pd.DataFrame
        Dataframe that contains the label 'premature'.
    df_static_information : pd.DataFrame
        Dataframe that contains all static data of the rec ids.
    x_data_to_be_fitted : pd.DataFrame
        Dataframe that contains the data you want to use for data preprocessing such as standard scaling.
        In general, this should be x_train.
    x_data_to_be_transformed : pd.DataFrame
        Dataframe that contains the data you want to transform by using the Pipeline that is fitted on
        x_data_to_be_fitted. Can also be the same as x_data_to_be_fitted.
    columns_to_use : List[columns]
        Names of the features you want to use for modeling.
    feature_name : str
        Name of the feature you want to calculate. Either 'sample_entropy', 'peak_frequency' or 'median_frequency'.
    reduced_seq_length : int
        The time window length of which you want to calculate feature_name on each time step.
        For example, if reduced_seq_length is 50 and feature_name is sample entropy, then you'll end up with
        50 values of the sample entropy which are calculated over non-overlapping time windows from df_signals_new.
    sub_seq_length : int
        The length of the sub-sequence you want to split reduced_seq_length into. For example, if reduced_seq_length
        is 50 and sub_seq_length is 10, then you'll end up with 5 sub-sequences that all have a length of 10.
    fs : int
        Sampling frequency of df_signals_new.
    shuffle : Boolean
        Whether to shuffle the DataLoader. Default is False. Shuffling is recommended during training.
    add_static_data : Boolean
        Default False. Whether to add the static data in the modelling.
    test_phase : bool
        Whether you want to generate a dataloader with the test data. If the case, then also the
        rec_ids will be returned.

    Returns
    -------
    [custom_loader_orig] : List[DataLoader]

    or

    [custom_loader_orig_part1, custom_loader_orig_part2] : List[DataLoader, DataLoader]

    List containing the dataloader(s).

    or if in test_phase : [custom_loader_orig_part1, custom_loader_orig_part2], rec_id_list :
    List[DataLoader, DataLoader], List[int].
    """
    df_feature = calculate_feature_over_fixed_time_window(df_signals_new,
                                                          fixed_seq_length=reduced_seq_length,
                                                          column_list=columns_to_use,
                                                          feature_name=feature_name,
                                                          fs=fs)

    df_feature = df_feature.merge(df_clinical_information[[c.REC_ID_NAME, 'premature']], how='left', on=c.REC_ID_NAME)

    if feature_name == 'sample_entropy':
        df_feature = replace_inf_with_zero(df_feature)
        df_feature = replace_na_with_zero(df_feature)

    # This df contains the feature data for the rec ids in x_data_to_be_transformed
    x_feature_transform = df_feature[
        df_feature[c.REC_ID_NAME].isin(x_data_to_be_transformed[c.REC_ID_NAME].unique())].reset_index(drop=True).copy()
    x_feature_fit = df_feature[df_feature[c.REC_ID_NAME].isin(x_data_to_be_fitted[c.REC_ID_NAME].unique())].reset_index(
        drop=True).copy()

    # To shuffle the rec ids in the batches during training
    if shuffle:
        recid_groups = [df for _, df in x_feature_transform.groupby(c.REC_ID_NAME)]
        random.shuffle(recid_groups)
        x_feature_transform = pd.concat(recid_groups).reset_index(drop=True)

        # Make sure that the rec ids in df_static_information are sorted exactly the same as in x_feature_transform
        df_static_information = df_static_information.set_index(c.REC_ID_NAME)
        df_static_information = df_static_information.reindex(index=x_feature_transform[c.REC_ID_NAME].unique())
        df_static_information = df_static_information.reset_index()

    # We need to re-do this step because the X dataframes now contain the label column ('premature')
    x_feature_transform, y_feature_transform = feature_label_split(x_feature_transform, 'premature')
    x_feature_fit, y_feature_fit = feature_label_split(x_feature_fit, 'premature')

    sequence_length_rec_ids_transform = x_feature_transform \
        .groupby(c.REC_ID_NAME, sort=False) \
        .agg({c.REC_ID_NAME: 'size'}) \
        .rename(columns={c.REC_ID_NAME: 'count'}) \
        .reset_index()

    # We keep the rec ids order to check later on if the data processing was consequent
    rec_ids_transform = list(sequence_length_rec_ids_transform[c.REC_ID_NAME])

    # The number of sub-sequences needed to make up an original sequence
    num_sub_sequences_fixed = int(reduced_seq_length / sub_seq_length)

    rec_ids_list = np.repeat(rec_ids_transform, num_sub_sequences_fixed)
    batch_lengths = np.repeat([sub_seq_length], len(rec_ids_transform) * num_sub_sequences_fixed)

    batch_length = pd.concat([pd.DataFrame(rec_ids_list, columns=[c.REC_ID_NAME]),
                              pd.DataFrame(batch_lengths, columns=['batch_timesteps'])], axis=1)

    print(batch_length)

    if add_static_data:
        # The static data will be processed (i.e., standard scaled, removing highly correlated features) and
        # added to the signals data
        # x_arr_static_fit, _, selected_columns_fit_static, rec_id_list_static_fit, _ = preprocess_static_data(
        #     df_static_information,
        #     x_feature_fit,
        #     x_feature_fit,
        #     threshold_correlation=0.85)
        #
        # x_arr_static_transform, _, _, _, rec_id_list_static_transform = preprocess_static_data(
        #     df_static_information,
        #     x_feature_fit,
        #     x_feature_transform,
        #     threshold_correlation=0.85)
        #
        # # Safety check if the data consequently processed for the correct order of rec ids
        # assert np.logical_and((np.array(rec_id_list_static_transform) == np.array(rec_ids_transform)).all(),
        #                       (np.array(rec_ids_transform) == np.array(x_feature_transform[c.REC_ID_NAME].unique())).all()), \
        #     f'Rec ids of x_feature_transform inconsequentially sorted during preprocessing!'
        #
        # df_static_fit = pd.concat([pd.DataFrame(rec_id_list_static_fit, columns=[c.REC_ID_NAME]),
        #                            pd.DataFrame(x_arr_static_fit, columns=selected_columns_fit_static)], axis=1)
        # df_static_transform = pd.concat([pd.DataFrame(rec_id_list_static_transform, columns=[c.REC_ID_NAME]),
        #                                  pd.DataFrame(x_arr_static_transform, columns=selected_columns_fit_static)],
        #                                 axis=1)

        x_feature_fit = x_feature_fit.merge(df_static_train, on=c.REC_ID_NAME).reset_index(drop=True)
        x_feature_transform = x_feature_transform.merge(df_static_test, on=c.REC_ID_NAME).reset_index(drop=True)

    # Drop the rec_id column from the X dataframe
    column_drop_pipeline = Pipeline([("columnDropper", ColumnDropperTransformer([c.REC_ID_NAME]))])

    # apply the column drop pipeline to dataframe
    x_arr_feature_transformed = fit_and_transform_data(x_feature_fit, x_feature_transform, column_drop_pipeline)
    x_arr_feature_fitted = fit_and_transform_data(x_feature_fit, x_feature_fit, column_drop_pipeline)

    total_cols = columns_to_use + selected_columns_static_data

    # ColumnTransformer needs a pd.DataFrame if columns are provided so we re-convert to a pd.DataFrame
    x_feature_fitted = pd.DataFrame(data=x_arr_feature_fitted, index=np.arange(len(x_arr_feature_fitted)),
                                    columns=[col for col in total_cols if col != c.REC_ID_NAME])

    x_feature_transformed = pd.DataFrame(data=x_arr_feature_transformed,
                                         index=np.arange(len(x_arr_feature_transformed)),
                                         columns=[col for col in total_cols if col != c.REC_ID_NAME])

    # set up pipeline for the signal data
    numeric_standard_pipe = Pipeline([('standard_scaler', StandardScaler())])

    # Set up columnTransformer
    # If using 'passthrough' for the remainder parameter, the columns when performing fit and transform must be in
    # the exact same order. We use the static_transformer only for the signals data (present in columns_to_use)
    static_transformer = ColumnTransformer(transformers=[('num_signal_vars', numeric_standard_pipe, columns_to_use)],
                                           remainder='passthrough')

    x_arr_feature_transformed = fit_and_transform_data(x_feature_fitted, x_feature_transformed,
                                                       static_transformer)

    y_arr_feature_transformed = fit_and_transform_data(y_feature_fit.values.ravel(),
                                                       y_feature_transform.values.ravel(), LabelEncoder())

    x_arr_feature_transformed, y_arr_feature_transformed, batch_sizes = \
        custom_sort_for_stateful_lstm(x_feature_transform, x_arr_feature_transformed, y_arr_feature_transformed,
                                      fixed_seq_length=reduced_seq_length, sub_seq_length=sub_seq_length,
                                      batch_size=params['batch_size'],
                                      num_features=len(columns_to_use) + len(selected_columns_static_data))

    # If the batch sizes are not equal across all batches we create two separate dataloaders
    if not batch_sizes.count(batch_sizes[0]) == len(batch_sizes):
        custom_loader_orig_part1, custom_loader_orig_part2, _, _ = \
            create_dataloader_if_batch_sizes_unequal(x_arr_feature_transformed, y_arr_feature_transformed,
                                                     params['batch_size'], batch_sizes, num_sub_sequences_fixed,
                                                     batch_length,
                                                     features_to_use=columns_to_use + selected_columns_static_data,
                                                     fixed_seq_length=reduced_seq_length,
                                                     sub_seq_length=sub_seq_length)

        if test_phase:
            return [custom_loader_orig_part1, custom_loader_orig_part2], rec_ids_transform

        elif not test_phase:
            return [custom_loader_orig_part1, custom_loader_orig_part2]

    if batch_sizes.count(batch_sizes[0]) == len(batch_sizes):
        custom_loader_orig, _ = create_dataloader_if_batch_sizes_equal(x_arr_feature_transformed,
                                                                       y_arr_feature_transformed, params['batch_size'],
                                                                       batch_length,
                                                                       features_to_use=columns_to_use + selected_columns_static_data,
                                                                       sub_seq_length=sub_seq_length)

        if test_phase:
            return [custom_loader_orig], rec_ids_transform
        elif not test_phase:
            return [custom_loader_orig]


def create_dataloader_if_batch_sizes_unequal(x_sorted: Union[pd.DataFrame, np.array],
                                             y_sorted: Union[pd.DataFrame, np.array],
                                             original_batch_size: int,
                                             batch_sizes: List[int],
                                             num_sub_sequences: int,
                                             df_batch_timesteps: pd.DataFrame,
                                             features_to_use: List[str],
                                             fixed_seq_length,
                                             sub_seq_length: int):
    """Create data loaders if the batch size are not equal across all batches.

    For instance, if the batch_sizes are [50, 50, 50, 30], then we create a dataloader for all
    batches with a size of 50, and a separate dataloader for the batch with size 30.

    To accomplish this we need to split the data in x_sorted and y_sorted accordingly.

    Parameters
    ----------
    x_sorted : pd.DataFrame
        Dataframe that contains the sorted features data. Result of custom_sort_for_stateful_lstm.
    y_sorted : pd.DataFrame
        Dataframe that contains the sorted targets. Result of custom_sort_for_stateful_lstm.
    original_batch_size : int
        The original batch size that was used. If not all rec ids fitted in an original batch size,
        then the remaining rec ids are all put in the last batch and this batch will have a different size.
    batch_sizes : List[int]
        List with the 'true' batch sizes used. For instance, if there are 180 instances and the original_batch_size
        is 50, then batch_sizes is [50, 50, 50, 30].
    num_sub_sequences : int
        Number of sub-sequences necessary to process an entire sequence.
    df_batch_timesteps : pd.DataFrame
        Dataframe that contains the number of non-padded timesteps for each rec id per batch.
        Result of compute_non_padded_timesteps_per_batch_of_original_seq.
    features_to_use : List[str]
        List with the names of the features you want to use for modeling.
    fixed_seq_length : int
        Length to which each sequence is padded/truncated to.
    sub_seq_length : int
        Length of the sub-sequences.

    Returns
    -------
    custom_loader_orig_part1, custom_loader_orig_part2, batch_lengths_part1, batch_lengths_part2 :
    Tuple[DataLoader, DataLoader, List[torch.Tensor], List[torch.Tensor]]
    """
    # Compute the number of rec ids that have been processed with a true_batch_size and
    # compute the number of rec ids that were left over
    num_processed_rec_ids_equal_batch_size = np.sum(
        [batch_size for batch_size in batch_sizes if batch_size == original_batch_size])
    num_processed_rec_ids_unequal_batch_size = np.sum(
        [batch_size for batch_size in batch_sizes if batch_size != original_batch_size])
    num_processed_rec_ids_unequal_batch_size = int(num_processed_rec_ids_unequal_batch_size)

    assert num_processed_rec_ids_unequal_batch_size != 0, f'There are ' \
                                                          f'{num_processed_rec_ids_unequal_batch_size} rec_ids ' \
                                                          f'that are part of an unequal batch size while this should ' \
                                                          f'have been 0!'

    # Split the X and y dataframes accordingly
    x_sorted_part1 = x_sorted[0:(num_processed_rec_ids_equal_batch_size * fixed_seq_length)].copy()
    x_sorted_part2 = x_sorted[(num_processed_rec_ids_equal_batch_size * fixed_seq_length):].copy()
    y_sorted_part1 = y_sorted[0:(num_processed_rec_ids_equal_batch_size * fixed_seq_length)].copy()
    y_sorted_part2 = y_sorted[(num_processed_rec_ids_equal_batch_size * fixed_seq_length):].copy()

    # Also split the df that contains the 'true' (non-padded) sequence length of each rec id
    df_batch_timesteps_x_part1 = df_batch_timesteps[0:(num_sub_sequences * num_processed_rec_ids_equal_batch_size)].copy()
    df_batch_timesteps_x_part2 = df_batch_timesteps[
                                 (num_sub_sequences * num_processed_rec_ids_equal_batch_size):].reset_index(drop=True).copy()

    assert (len(y_sorted_part1) + len(y_sorted_part2)) == len(
        y_sorted), f'Length of splitted dfs summed is {len(y_sorted_part1) + len(y_sorted_part2)}, ' \
                   f'but should be {len(y_sorted)}!'
    assert (len(df_batch_timesteps_x_part1) + len(df_batch_timesteps_x_part2)) == len(df_batch_timesteps), \
        f'Length of splitted batch lengths dfs summed is ' \
        f'{len(df_batch_timesteps_x_part1) + len(df_batch_timesteps_x_part2)}, but should be {len(df_batch_timesteps)}!'

    df_custom_sort_lengths_x_part1 = custom_sort_seq_lengths(df_batch_timesteps_x_part1,
                                                             batch_size=original_batch_size)
    df_custom_sort_lengths_x_part2 = custom_sort_seq_lengths(df_batch_timesteps_x_part2,
                                                             batch_size=num_processed_rec_ids_unequal_batch_size)

    batch_lengths_part1 = torch.LongTensor(list(df_custom_sort_lengths_x_part1['batch_timesteps'].values))
    batch_lengths_part2 = torch.LongTensor(list(df_custom_sort_lengths_x_part2['batch_timesteps'].values))

    features_orig_part1 = torch.from_numpy(x_sorted_part1)
    features_orig_part2 = torch.from_numpy(x_sorted_part2)

    targets_orig_part1 = torch.from_numpy(y_sorted_part1).float().view(-1, 1)
    targets_orig_part2 = torch.from_numpy(y_sorted_part2).float().view(-1, 1)

    # Put in TensorDataset format
    dataset_orig_part1 = TensorDataset(features_orig_part1, targets_orig_part1)
    dataset_orig_part2 = TensorDataset(features_orig_part2, targets_orig_part2)

    # At this moment, each time step is a separate dedicated tensor. We want to merge
    # together the time steps in chunks of 200 time steps. The time steps in a chunk must
    # have no overlap with other chunks. The features will have the shape
    # [sub_seq_length (=200), num_features] and the target will have the shape [num_targets].
    # Create custom Datasets with non-overlapping sequences of length 200
    custom_non_overlap_dataset_orig_part1 = NonOverlappingSequencesDataset(dataset_orig_part1,
                                                                           sub_seq_length=sub_seq_length,
                                                                           num_features=len(features_to_use),
                                                                           num_targets=1, start_point=0)
    custom_non_overlap_dataset_orig_part2 = NonOverlappingSequencesDataset(dataset_orig_part2,
                                                                           sub_seq_length=sub_seq_length,
                                                                           num_features=len(features_to_use),
                                                                           num_targets=1, start_point=0)

    # We now create a DataLoader object that wraps an iterable around the Dataset to enable easy access
    # to the samples. The DataLoader features will have shape [batch_size, sub_seq_length, num_features]
    # and the target will have shape [batch_size, num_targets]

    # Please note that the batch_size must be <= len(data) / sub_seq_length. Otherwise, there are too few
    # datapoints to fit in the number of batches.
    # The order MUST be preserved, so do not shuffle!
    custom_loader_orig_part1 = DataLoader(custom_non_overlap_dataset_orig_part1,
                                          batch_size=original_batch_size,
                                          shuffle=False, drop_last=False)

    custom_loader_orig_part2 = DataLoader(custom_non_overlap_dataset_orig_part2,
                                          batch_size=num_processed_rec_ids_unequal_batch_size,
                                          shuffle=False, drop_last=False)

    return custom_loader_orig_part1, custom_loader_orig_part2, batch_lengths_part1, batch_lengths_part2


def create_dataloader_if_batch_sizes_equal(x_sorted: np.array, y_sorted: np.array,
                                           batch_size: int,
                                           df_batch_timesteps: pd.DataFrame,
                                           features_to_use: List[str],
                                           sub_seq_length: int):
    """Create data loaders if the batch sizes are equal across all batches.

    For instance, if the batch_sizes are [60, 60, 60], then we create one dataloader for all
    batches.

    Parameters
    ----------
    x_sorted : np.array
        Array that contains the sorted features data. Result of custom_sort_for_stateful_lstm.
    y_sorted : np.array
        Array that contains the sorted targets. Result of custom_sort_for_stateful_lstm.
    batch_size : int
        The original batch size that was used.
    df_batch_timesteps : pd.DataFrame
        Dataframe that contains the number of non-padded timesteps for each rec id per batch.
        Result of compute_non_padded_timesteps_per_batch_of_original_seq.
    features_to_use : List[str]
        List with the names of the features to use for modeling.
    sub_seq_length : int
        Length of each sub-sequence.

    Returns
    -------
    custom_loader_orig, batch_lengths : Tuple[DataLoader, List[torch.Tensor]]
    """
    df_custom_sort_lengths_x = custom_sort_seq_lengths(df_batch_timesteps, batch_size=batch_size)

    batch_lengths = torch.LongTensor(list(df_custom_sort_lengths_x['batch_timesteps'].values))

    features_orig = torch.from_numpy(x_sorted)
    targets_orig = torch.from_numpy(y_sorted).float().view(-1, 1)

    # Put in TensorDataset format
    dataset_orig = TensorDataset(features_orig, targets_orig)

    # At this moment, each time step is a separate dedicated tensor. We want to merge
    # together the time steps in chunks of 200 time steps. The time steps in a chunk must
    # have no overlap with other chunks. The features will have the shape
    # [sub_seq_length (=200), num_features] and the target will have the shape [num_targets].
    # Create custom Datasets with non-overlapping sequences of length 200
    custom_non_overlap_dataset_orig = NonOverlappingSequencesDataset(dataset_orig, sub_seq_length=sub_seq_length,
                                                                     num_features=len(features_to_use),
                                                                     num_targets=1, start_point=0)

    # We now create a DataLoader object that wraps an iterable around the Dataset to enable easy access
    # to the samples. The DataLoader features will have shape [batch_size, sub_seq_length, num_features]
    # and the target will have shape [batch_size, num_targets]

    # Please note that the batch_size must be <= len(data) / sub_seq_length. Otherwise, there are too few
    # datapoints to fit in the number of batches.
    # The order MUST be preserved, so do not shuffle!
    custom_loader_orig = DataLoader(custom_non_overlap_dataset_orig, batch_size=batch_size,
                                    shuffle=False, drop_last=False)

    return custom_loader_orig, batch_lengths


def fit_and_transform_data(data_to_be_fitted: Union[pd.DataFrame, np.array],
                           data_to_be_transformed: Union[pd.DataFrame, np.array], pipeline) -> np.array:
    # apply the pipeline to dataframe
    fitted_pipeline = pipeline.fit(data_to_be_fitted)
    data_to_be_transformed_arr = fitted_pipeline.transform(data_to_be_transformed)

    return data_to_be_transformed_arr


def create_tensordataset_reduced_seq_length(df_signals_new: pd.DataFrame,
                                            x_data_to_be_fitted: pd.DataFrame,
                                            x_data_to_be_transformed: pd.DataFrame,
                                            df_clinical_information: pd.DataFrame,
                                            optimal_params: Dict, columns_to_use: List[str],
                                            feature_name: str, fs: int):
    """Create a custom TensorDataset by reducing the sequence length to the optimal hyperparameter
    of reduced_seq_length. This is done by calculating a feature (sample entropy/peak freq/median freq)
    over df_signals_new.

    Also, the data will be standard scaled by fitting it on x_data_to_be_fitted and apply it on
    x_data_to_be_transformed.

    Parameters
    ----------
    df_signals_new : pd.DataFrame
        DataFrame that contains the signal data after applying remove_first_n_samples_of_signals and
        remove_last_n_samples_of_signals.
    x_data_to_be_fitted : pd.DataFrame
        Dataframe that contains the data you want to use for data preprocessing such as standard scaling.
        In general, this should be x_train.
    x_data_to_be_transformed : pd.DataFrame
        Dataframe that contains the data you want to transform by using the Pipeline that is fitted on
        x_data_to_be_fitted. Can also be the same as x_data_to_be_fitted.
    df_clinical_information : pd.DataFrame
        Dataframe that contains the label 'premature'.
    optimal_params : Dict
        Dictionary containing the optimal hyperparameters and its values.
    columns_to_use : List[columns]
        Names of the columns you want to use for modeling.
    feature_name : str
        Name of the feature you want to use to compute over the columns in columns_to_use in df_signals_new.
        Either 'sample_entropy', 'peak_frequency' or 'median_frequency'.
    fs : int
        Sampling frequency in df_signals_new.

    Returns
    -------
    custom_non_overlap_dataset_samp_en : TensorDataset
        TensorDataset with non-overlapping sequences of length reduced_seq_length for each rec_id.
    """
    df_feature = calculate_feature_over_fixed_time_window(df_signals_new,
                                                          fixed_seq_length=optimal_params['reduced_seq_length'],
                                                          column_list=columns_to_use,
                                                          feature_name=feature_name,
                                                          fs=fs)

    df_feature = df_feature.merge(df_clinical_information[[c.REC_ID_NAME, 'premature']], how='left', on=c.REC_ID_NAME)

    if feature_name == 'sample_entropy':
        df_feature = replace_inf_with_zero(df_feature)
        df_feature = replace_na_with_zero(df_feature)

    # This df contains the feature (samp en/peak freq/median freq) data for the rec ids in x_data_to_be_transformed
    x_feature_transform = df_feature[df_feature[c.REC_ID_NAME].isin(x_data_to_be_transformed[c.REC_ID_NAME].unique())].reset_index(drop=True).copy()
    x_feature_fit = df_feature[df_feature[c.REC_ID_NAME].isin(x_data_to_be_fitted[c.REC_ID_NAME].unique())].reset_index(drop=True).copy()

    # We need to re-do this step because the X dataframes now contain the label column ('premature')
    x_feature_transform, y_feature_transform = feature_label_split(x_feature_transform, 'premature')
    x_feature_fit, y_feature_fit = feature_label_split(x_feature_fit, 'premature')

    # Drop the rec_id column from the X dataframe
    column_drop_pipeline = Pipeline([("columnDropper", ColumnDropperTransformer([c.REC_ID_NAME]))])

    # apply the pipeline to dataframe
    x_arr_feature_transformed = fit_and_transform_data(x_feature_fit, x_feature_transform, column_drop_pipeline)
    x_arr_feature_fitted = fit_and_transform_data(x_feature_fit, x_feature_fit, column_drop_pipeline)

    standard_scalar_pipe = Pipeline([('standard_scaler', StandardScaler())])
    x_arr_feature_transformed = fit_and_transform_data(x_arr_feature_fitted, x_arr_feature_transformed,
                                                       standard_scalar_pipe)

    y_arr_feature_transformed = fit_and_transform_data(y_feature_fit.values.ravel(),
                                                       y_feature_transform.values.ravel(), LabelEncoder())

    # Create tensors of the X and y data. BCEWithLogitsLoss requires its target to be a float tensor, not long.
    features = torch.from_numpy(x_arr_feature_transformed)
    targets = torch.from_numpy(y_arr_feature_transformed).float().view(-1, 1)

    # We put the features and targets in TensorDataset format
    dataset = TensorDataset(features, targets)

    # Create custom Datasets with non-overlapping sequences of length reduced_seq_length
    custom_non_overlap_dataset = NonOverlappingSequencesDataset(dataset,
                                                                optimal_params['reduced_seq_length'],
                                                                len(columns_to_use), num_targets=1, start_point=0)

    return custom_non_overlap_dataset


def create_custom_data_loaders(trial, df_signals: pd.DataFrame, x: pd.DataFrame, features_to_use: List[str],
                               num_sub_sequences: int, params: Dict, shuffle=False):
    """Create data loaders from the original signal data.

    This function is for the specific use case in which we have a very long sequence which will be
    splitted into multiple sub-sequences. The order of the sub-sequences, however, MUST remain intact
    as otherwise processing makes no sense, i.e., you want to process the sub-sequences of each rec_id
    sequentially until all sub-sequences of a rec id have been processed and not mix sub-sequences of
    different rec ids. We therefore cannot make use of the shuffle function in PyTorch DataLoader and
    are therefore creating a custom method.

    The signal data from the rec_ids present in x will be selected and splitted into X and y dfs. If shuffle=True,
    the order of the rec ids will be shuffled and AFTER shuffling, the long sequence will be splitted
    into multiple sub-sequences. This results in shuffled dataloaders every time this function is used,
    but with the intact order of the sub-sequences per rec id.

    Furthermore, this function allows for unequal batch_sizes to process all rec ids. For instance, if there are
    180 rec ids present in X and the original batch size is 50, then there are [50, 50, 50, 30] batch sizes
    needed to process all rec ids. This function will create a separate dataloader for the rec ids that fit
    into a batch size of 50, and for the remaining 30 rec ids a separate dataloader will be created. When
    the model will be trained, it will be trained on these two dataloaders sequentially. This step
    is also necessary to ensure that the order of the sub-sequences for each rec id remains intact.

    Parameters
    ----------
    trial :

    df_signals : pd.DataFrame
        Dataframe that contains the original signal data.
    x : pd.DataFrame
        Dataframe containing the signals data for either the train/val/test data.
    features_to_use : List[str]
        List of the names of the columns you want to use for modeling.
    num_sub_sequences : int
        Number of sub-sequences necessary to process an entire sequence.
    params : Dict
        Dictionary containing the hyperparameters and its values.
    shuffle : Boolean
        To shuffle the rec_ids such that the dataloader that will be created contains different rec_ids.
        Is useful when one wants to create a train dataloader. For the val and test dataloader it is
        recommended to not shuffle the rec_ids.

    Returns
    -------
    [custom_loader_orig, batch_lengths] : List[DataLoader, List[torch.Tensor]]

    or

    [custom_loader_orig_part1, custom_loader_orig_part2, batch_sizes_x_part1, batch_sizes_x_part2] :
    List[DataLoader, DataLoader, List[torch.Tensor], List[torch.Tensor]]

    List containing the dataloader(s) and batch lengths
    """
    # This df contains the original signal data from the rec_ids present in x
    x_orig = df_signals[df_signals[c.REC_ID_NAME].isin(x[c.REC_ID_NAME].unique())].reset_index(drop=True).copy()

    if shuffle:
        recid_groups = [df for _, df in x_orig.groupby(c.REC_ID_NAME)]
        random.shuffle(recid_groups)
        x_orig = pd.concat(recid_groups).reset_index(drop=True)

    # We need to re-do this step because the X dataframes now contain the label column
    x_orig, y_orig = feature_label_split(x_orig, 'premature')

    # We drop these columns from X_orig as we won't be using these features during modeling
    cols_not_to_use = [col for col in df_signals.columns if col not in features_to_use]
    column_drop_pipeline = Pipeline([("columnDropper", ColumnDropperTransformer(cols_not_to_use))])

    x_arr_orig = column_drop_pipeline.fit_transform(x_orig)
    lb = LabelEncoder()
    y_arr_orig = lb.fit_transform(y_orig['premature'])

    # Pad all sequences to the same length
    x_arr_padded_orig, y_arr_padded_orig = pad_sequences_to_fixed_length(x_orig, x_arr_orig, y_arr_orig,
                                                                         num_features=len(features_to_use),
                                                                         fixed_seq_length=28800)

    # Custom sort x_arr_padded_orig and y_arr_padded_orig such that we can correctly batchify
    # (i.e., ensure that the sub-sequences are processed sequentially for each rec_id) the data
    # using the DataLoader class.
    x_arr_padded_sorted_orig, y_arr_padded_sorted_orig, batch_sizes_x = \
        custom_sort_for_stateful_lstm(x_orig, x_arr_padded_orig, y_arr_padded_orig, fixed_seq_length=28800,
                                      sub_seq_length=200, batch_size=params['batch_size'],
                                      num_features=len(features_to_use))

    df_batch_timesteps_x = compute_non_padded_timesteps_per_batch_of_original_seq(
        x_orig[[c.REC_ID_NAME] + features_to_use], fixed_seq_length=28800, sub_seq_length=200)

    # If the batch sizes are not equal across all batches we create two separate dataloaders
    if not batch_sizes_x.count(batch_sizes_x[0]) == len(batch_sizes_x):
        custom_loader_orig_part1, custom_loader_orig_part2, batch_sizes_x_part1, batch_sizes_x_part2 = \
            create_dataloader_if_batch_sizes_unequal(x_arr_padded_sorted_orig, y_arr_padded_sorted_orig,
                                                     params['batch_size'], batch_sizes_x, num_sub_sequences,
                                                     df_batch_timesteps_x, features_to_use, fixed_seq_length=28800,
                                                     sub_seq_length=200)

        return [custom_loader_orig_part1, custom_loader_orig_part2, batch_sizes_x_part1, batch_sizes_x_part2]

    if batch_sizes_x.count(batch_sizes_x[0]) == len(batch_sizes_x):
        custom_loader_orig, batch_lengths = create_dataloader_if_batch_sizes_equal(x_arr_padded_sorted_orig,
                                                                                   y_arr_padded_sorted_orig,
                                                                                   params['batch_size'],
                                                                                   df_batch_timesteps_x,
                                                                                   features_to_use,
                                                                                   sub_seq_length=200)

        return [custom_loader_orig, batch_lengths]


def generate_custom_data_loaders(trial, df_signals: pd.DataFrame, x: pd.DataFrame,
                                 features_to_use: List[str], num_sub_sequences: int,
                                 params: Dict, shuffle: bool = True) -> Tuple[List[DataLoader], List[torch.Tensor]]:
    """Create data loaders from df_signals with either an equal batch size across all batches
    or unequal batch sizes.

    Parameters
    ----------
    trial :

    df_signals : pd.DataFrame
        DataFrame that contains the original signal data.
    x : pd.DataFrame
        Dataframe that contains the data of either the train/val/test rec ids.
    features_to_use : List[columns]
        Names of the features you want to use for modeling.
    num_sub_sequences : int
        Number of sub-sequences necessary to process an entire sequence.
    params : Dict
        Dictionary containing the hyperparameters and its values.
    shuffle : Boolean
        Whether to shuffle the rec ids in X such that the data loader will have a different order of rec ids.

    Returns
    -------
    custom_loader_orig, batch_lengths : Tuple[List[DataLoader], List[torch.Tensor]]
        List with the dataloader(s) and batch sizes.
    """
    # Shuffle means shuffle the rec_ids at the beginning of every epoch while keeping the
    # order of the sub-sequences per rec id intact
    list_dls = create_custom_data_loaders(trial, df_signals, x, features_to_use,
                                          num_sub_sequences, params, shuffle=shuffle)

    # If the batch sizes are unequal across all batches, then two separate dataloaders have been created
    # Here we unpack the dataloaders
    if len(list_dls) > 2:
        custom_loader_orig_part1 = list_dls[0]
        custom_loader_orig_part2 = list_dls[1]

        custom_loader_orig = [custom_loader_orig_part1, custom_loader_orig_part2]

        batch_sizes_x_part1 = list_dls[2]
        batch_sizes_x_part2 = list_dls[3]

        batch_sizes_x = torch.cat((batch_sizes_x_part1, batch_sizes_x_part2), dim=0)

    # If the batch sizes are equal across all batches, then one dataloader has been created
    if len(list_dls) == 2:
        custom_loader_orig = [list_dls[0]]
        batch_sizes_x = list_dls[1]

    return custom_loader_orig, batch_sizes_x


def generate_feature_data_loaders(trial, df_signals: pd.DataFrame, df_clinical_information: pd.DataFrame,
                                  df_static_information: pd.DataFrame,
                                  x_data_to_be_fitted: pd.DataFrame, x_data_to_be_transformed: pd.DataFrame,
                                  x_train_static_data: pd.DataFrame, x_test_static_data: pd.DataFrame,
                                  selected_columns_static_data: List[str],
                                  params: Dict, columns_to_use: List[str], feature_name: str, reduced_seq_length: int,
                                  sub_seq_length: int, fs: int, shuffle: bool = True,
                                  add_static_data: bool = False, test_phase: bool = False) -> List[DataLoader]:
    """Create data loaders from df_signals with either an equal batch size across all batches
    or unequal batch sizes.

    Parameters
    ----------
    trial :

    df_signals : pd.DataFrame
        DataFrame that contains the original signal data.
    df_clinical_information : pd.DataFrame
        Dataframe containing the target variable 'premature'.
    df_static_information : pd.DataFrame
        Dataframe that contains all static data belonging to the rec ids.
    x_data_to_be_fitted : pd.DataFrame
        Dataframe that contains the train data of rec ids. This data will be used to fit standard scaling etc.
        and then apply it on data_to_be_transformed.
    x_data_to_be_transformed : pd.DataFrame
        Dataframe that contains either the train/val/test data of rec ids.
        This data will be used create the final dataloader with.
    x_train_static_data : pd.DataFrame
        Dataframe that contains the static data for training.
    x_test_static_data : pd.DataFrame
        Dataframe that contains the static data for testing.
    selected_columns_static_data : List[str]
        Columns to use of the static data.
    columns_to_use : List[columns]
        Names of the columns you want to use for modeling.
    feature_name : str
        Name of the feature you want to calculate over df_signals. Must be either 'sample_entropy', 'peak_frequency' or
        'median_frequency'.
    reduced_seq_length : int
        The number of time steps you want to reduce the signals in df_signals to (per rec id). The feature_name will
        be calculated over reduced_seq_length time windows.
    sub_seq_length : int
        The number of time steps you want to use to split reduced_seq_length into. For example, if reduced_seq_length
        is 50 and sub_seq_length is 10, then you'll have 5 sub-sequences that make up the total reduced_seq_length.
    fs : int
        Sampling frequency in df_signals.
    params : Dict
        Dictionary containing the hyperparameters and its values.
    shuffle : Boolean
        Whether to shuffle the rec ids in x_data_to_be_transformed such that the data loader will have a
        different order of rec ids.
    add_static_data : Boolean
        Default is False. Whether to add the static data for modelling.
    test_phase : bool
        Whether you want to generate a dataloader with the test data. If the case, then also the
        rec_ids will be returned.

    Returns
    -------
    custom_loader_orig : List[DataLoader(s)]
        List with the dataloader(s).
    """
    # Shuffle means shuffle the rec_ids at the beginning of every epoch while keeping the
    # order of the sub-sequences per rec id intact
    list_dls = custom_calculate_feature_over_splited_fixed_time_window(trial, params, df_signals,
                                                                       df_clinical_information, df_static_information,
                                                                       x_data_to_be_fitted,
                                                                       x_data_to_be_transformed,
                                                                       x_train_static_data,
                                                                       x_test_static_data,
                                                                       selected_columns_static_data,
                                                                       columns_to_use,
                                                                       feature_name, reduced_seq_length, sub_seq_length,
                                                                       fs, shuffle=shuffle,
                                                                       add_static_data=add_static_data,
                                                                       test_phase=test_phase)

    # If the batch sizes are equal across all batches, then one dataloader has been created
    if len(list_dls) == 1:
        custom_loader_orig = [list_dls[0]]

    # If the batch sizes are unequal across all batches, then two separate dataloaders have been created
    # Here we unpack the dataloaders
    if len(list_dls) == 2:
        custom_loader_orig_part1 = list_dls[0]
        custom_loader_orig_part2 = list_dls[1]

        custom_loader_orig = [custom_loader_orig_part1, custom_loader_orig_part2]

    return custom_loader_orig


def create_final_train_and_test_feature_dataloaders(df_signals_new: pd.DataFrame,
                                                    x_train: pd.DataFrame,
                                                    x_val: pd.DataFrame, x_test: pd.DataFrame,
                                                    df_clinical_information: pd.DataFrame, optimal_params: Dict,
                                                    columns_to_use: List[str], feature_name: str, fs: int) -> \
        Tuple[DataLoader, DataLoader]:
    """Create data loaders from df_signals_new for the final train of the model. The train and val data
    will be combined into one dataset and one dataloader will be created for the combined dataset. Also,
    a dataloader for the test dataset will be created.

    Parameters
    ----------
    df_signals_new : pd.DataFrame
        DataFrame that contains the signal data after applying remove_first_n_samples_of_signals and
        remove_last_n_samples_of_signals.
    x_train : pd.DataFrame
        Dataframe that contains the train data.
    x_val : pd.DataFrame
        Dataframe that contains the validation data.
    x_test : pd.DataFrame
        Dataframe that contains the test data.
    df_clinical_information : pd.DataFrame
        Dataframe that contains the label 'premature'.
    optimal_params : Dict
        Dictionary containing the optimal hyperparameters and its values.
    columns_to_use : List[columns]
        Names of the columns you want to use for modeling.
    feature_name : str
        Name of the feature which you wat to compute from df_signals_new. Either 'sample_entropy', 'peak_frequency' or
        'median_frequency'.
    fs : int
        Sampling frequency of df_signals_new.

    Returns
    -------
    train_val_loader_samp_en, test_loader_samp_en : Tuple[DataLoader, DataLoader]
        Dataloaders with the combined train + val data in one dataloader and the test data in the other dataloader.
    """
    custom_non_overlap_train_dataset = create_tensordataset_reduced_seq_length(df_signals_new, x_train, x_train,
                                                                               df_clinical_information,
                                                                               optimal_params, columns_to_use,
                                                                               feature_name=feature_name, fs=fs)

    custom_non_overlap_val_dataset = create_tensordataset_reduced_seq_length(df_signals_new, x_train, x_val,
                                                                             df_clinical_information,
                                                                             optimal_params, columns_to_use,
                                                                             feature_name=feature_name, fs=fs)

    custom_non_overlap_test_dataset = create_tensordataset_reduced_seq_length(df_signals_new, x_train, x_test,
                                                                              df_clinical_information,
                                                                              optimal_params, columns_to_use,
                                                                              feature_name=feature_name, fs=fs)

    # For the final evaluation we do a final train on the train and val dataset combined and
    # then test on the test set
    train_val = ConcatDataset([custom_non_overlap_train_dataset, custom_non_overlap_val_dataset])

    # Create one dataloader with the train and val data combined for the final training
    train_val_loader = DataLoader(train_val, batch_size=optimal_params['batch_size'],
                                  shuffle=True, drop_last=False)

    # We do not shuffle the samples in the test_loader
    test_loader = DataLoader(custom_non_overlap_test_dataset,
                             batch_size=optimal_params['batch_size'],
                             shuffle=False, drop_last=False)

    return train_val_loader, test_loader
