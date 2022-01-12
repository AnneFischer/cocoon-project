from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd
from typing import List, Tuple, Any
from sklearn.model_selection import train_test_split
import constants as c
import math
import torch
from torch.utils.data import TensorDataset, Dataset
import antropy as ant


def butter_bandpass_filter(data: np.array, low_cut: np.float, high_cut: np.float,
                           fs: np.float, order: int, axis: int = 0):
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

        df_tmp_rec_id = df_signals[df_signals[c.REC_ID_NAME] == rec_id]
        df_filtered_signals = pd.DataFrame(df_tmp_rec_id[[c.REC_ID_NAME]], columns=[c.REC_ID_NAME])

        for channel in list_of_channels:
            # The original signal data is also added to the final dataframe
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
    df_signals['time_step'] = df_signals.groupby(c.REC_ID_NAME).cumcount()

    print(f'The number of data points before removing the first {n} data points (per rec_id) is: {len(df_signals)}')

    df_signals_new = df_signals[(df_signals['time_step'] >= n)].reset_index(drop=True)

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
                                          .values)].reset_index(drop=True)

    print(f'The number of data points after removing the last {n} data points (per rec_id) is: {len(df_signals_new)}')

    return df_signals_new


def calculate_samp_en_over_fixed_time_window(df_signals: pd.DataFrame, fixed_seq_length: int,
                                             feature_list: List[str]) -> pd.DataFrame:
    """Calculate the sample entropy over fixed_seq_length time windows. In effect, you reduce the total sequence
    length to fixed_seq_length.

    Example: If the total sequence length of a rec_id is 27790 time steps and the fixed_seq_length is set at 300
    time steps, then the sample entropy will be calculated over 300 consecutive subsequences. Each subsequence will have
    approximately the same length (and in this case the total length of the subsequences sums up to 27790). Thus, the
    subsequences over which the sample entropy is calculated will not have had the exact same length.

    Parameters
    ----------
    df_signals : pd.DataFrame
        Original signal data of the features. Does not include the target. Must contain the column 'rec_id'.
    fixed_seq_length : int
        Universal length you want to truncate the sequence of each rec_id to.
    feature_list : List[str]
        List with the names of the features from which you want to calculate the sample entropy.

    Returns
    -------
    df_samp_en : pd.DataFrame
        Dataframe that contains the sample entropy values for each subsequence for each rec_id. This dataframe
        will contain num_rec_ids*fixed_seq_length rows.
    """
    num_rec_ids = df_signals[c.REC_ID_NAME].nunique()
    num_features = len(feature_list)

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
    # - Each feature
    # - Each subsequence
    # To calculate the sample entropy for each rec_id, for each feature and for each subsequence
    for i, rec_id in enumerate(sequence_length_rec_ids[c.REC_ID_NAME]):
        rec_id = int(rec_id)
        rec_id_samp_en = []

        for j, feature in enumerate(feature_list):
            # array_split splits an array into multiple sub-arrays of approx the same length.
            # We use fixed_seq_length as the number of sub-arrays
            chunked_rec_id = np.array_split(df_signals.loc[df_signals[c.REC_ID_NAME] == rec_id, [feature]],
                                            fixed_seq_length)
            feature_samp_en = []

            for i in range(len(chunked_rec_id)):
                samp_en_chunk = ant.sample_entropy(chunked_rec_id[i].to_numpy().flatten())
                feature_samp_en.append(samp_en_chunk)
            rec_id_samp_en.append(feature_samp_en)
        # samp_en_arr_rec_id contains the sample entropy for all features and each feature has its own column
        samp_en_arr_rec_id = np.column_stack(rec_id_samp_en)
        # Here we add the rec_id as a column
        samp_en_arr_rec_id = np.concatenate((np.array([rec_id] * fixed_seq_length).reshape(fixed_seq_length, -1),
                                             samp_en_arr_rec_id), axis=1)

        x_arr_final[start_index:(start_index + fixed_seq_length)] = samp_en_arr_rec_id
        start_index += fixed_seq_length

    df_samp_en = pd.DataFrame(x_arr_final, columns=['rec_id'] + feature_list)
    df_samp_en['rec_id'] = df_samp_en['rec_id'].astype('int64')

    return df_samp_en


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
    df_label = df[[target_col]]
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

    return x_train, x_val, x_test, y_train, y_val, y_test


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
        .reset_index()

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
            # This label is the same throughout the entire sequence so we can take any value between
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


def pad_sequences_to_fixed_length_df(df_features: pd.DataFrame, num_features: int = 12, fixed_seq_length: int = 28800):
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
        .reset_index()

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
                                  num_features: int = 12) -> Tuple[np.ndarray, np.ndarray]:
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

    for rec_id_batch in rec_id_batches:
        for block in range(0, time_step_blocks):
            for rec_id in range(rec_id_batch, rec_id_batch + batch_size):
                # sorted_indexes contains the indexes in the correct order for the batch_size rec_ids for the block time
                # steps. We will use sorted_indexes to extract the values in the correct order from X_arr_padded and
                # y_arr_padded
                sorted_indexes = [x for x in range((rec_id * fixed_seq_length) + block * sub_seq_length,
                                                   (rec_id * fixed_seq_length) + (block + 1) * sub_seq_length)]
                new_indexes_order.append(sorted_indexes)

    flattened_list = list(np.array(new_indexes_order).flat)

    x_sorted[0:(num_rec_ids * fixed_seq_length)] = x_arr_padded[flattened_list]
    y_sorted[0:(num_rec_ids * fixed_seq_length)] = y_arr_padded[flattened_list]

    return x_sorted, y_sorted


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

