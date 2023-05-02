import pandas as pd
import sys
from src_pre_term_database.optimization import OptimizationTCNFeatureSequence, \
    OptimizationStatefulFeatureSequenceLSTM, OptimizationCombinedLSTM, OptimizationTCNFeatureSequenceCombined, \
    OptimizationTCNFeatureSequenceCombinedCopies
from src_pre_term_database.modeling import TCN, LSTMStatefulClassificationFeatureSequence, \
    LSTMCombinedModel, TCNCombinedModel, TCNCombinedModelCopies
from src_pre_term_database.load_dataset import build_clinical_information_dataframe, build_demographics_dataframe
from src_pre_term_database.data_processing_and_feature_engineering import train_val_test_split, \
    preprocess_static_data, add_static_data_to_signal_data, \
    basic_preprocessing_static_data, basic_preprocessing_signal_data, generate_dataloader, preprocess_signal_data, \
    feature_label_split
from src_pre_term_database.utils import read_settings
from src_pre_term_database.final_train import get_best_params, get_best_params_comb_model
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Union
import constants as c
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import math
import argparse
import os
import datetime
from sklearn.linear_model import LogisticRegression


settings_path = os.path.abspath("references/settings")

file_paths = read_settings(settings_path, 'file_paths')
data_path = file_paths['data_path']

optional_model_dict = {
    "lstm_sample_entropy_with_static_data_fold_0":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(30, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True), nn.ReLU())},
    "lstm_sample_entropy_with_static_data_fold_1":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(22, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True), nn.ReLU())},
    "lstm_sample_entropy_with_static_data_fold_2":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(21, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=21, out_features=13, bias=True), nn.ReLU())},
    "lstm_sample_entropy_with_static_data_fold_3":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(23, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True), nn.ReLU())},
    "lstm_sample_entropy_with_static_data_fold_4":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(21, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=21, out_features=20, bias=True), nn.ReLU())},

    "lstm_peak_frequency_with_static_data_fold_0":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(22, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU())},
    "lstm_peak_frequency_with_static_data_fold_1":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(31, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU())},
    "lstm_peak_frequency_with_static_data_fold_2":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU())},
    "lstm_peak_frequency_with_static_data_fold_3":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(35, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU())},
    "lstm_peak_frequency_with_static_data_fold_4":
        {'optional_model': nn.Sequential(nn.ReLU(), nn.Linear(in_features=28, out_features=17, bias=True), nn.ReLU())},

    "lstm_median_frequency_with_static_data_fold_0":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU())},
    "lstm_median_frequency_with_static_data_fold_1":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU())},
    "lstm_median_frequency_with_static_data_fold_2":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU())},
    "lstm_median_frequency_with_static_data_fold_3":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU())},
    "lstm_median_frequency_with_static_data_fold_4":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(31, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=31, out_features=20, bias=True), nn.ReLU())},
    "tcn_sample_entropy_with_static_data_fold_0":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(35, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=35, out_features=16, bias=True), nn.ReLU())},
    "tcn_sample_entropy_with_static_data_fold_1":
        {'optional_model': nn.Sequential(nn.ReLU(), nn.Linear(in_features=31, out_features=15, bias=True), nn.ReLU())},
    "tcn_sample_entropy_with_static_data_fold_2":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=28, out_features=13, bias=True), nn.ReLU())},
    "tcn_sample_entropy_with_static_data_fold_3":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=28, out_features=18, bias=True), nn.ReLU())},
    "tcn_sample_entropy_with_static_data_fold_4":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(31, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=31, out_features=14, bias=True), nn.ReLU())},


    "tcn_peak_frequency_with_static_data_fold_0":
        {'optional_model': nn.Sequential(nn.ReLU(), nn.Linear(in_features=31, out_features=15, bias=True), nn.ReLU())},
    "tcn_peak_frequency_with_static_data_fold_1":
        {'optional_model': nn.Sequential(nn.ReLU(), nn.Linear(in_features=32, out_features=15, bias=True), nn.ReLU())},
    "tcn_peak_frequency_with_static_data_fold_2":
        {'optional_model': nn.Sequential(nn.ReLU(), nn.Linear(in_features=29, out_features=14, bias=True), nn.ReLU())},
    "tcn_peak_frequency_with_static_data_fold_3":
        {'optional_model': nn.Sequential(nn.ReLU(), nn.Linear(in_features=31, out_features=14, bias=True), nn.ReLU())},
    "tcn_peak_frequency_with_static_data_fold_4":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=29, out_features=14, bias=True), nn.ReLU())},
    "tcn_median_frequency_with_static_data_fold_0": {'optional_model': nn.Sequential(nn.ReLU())},
    "tcn_median_frequency_with_static_data_fold_1":
        {'optional_model': nn.Sequential(nn.ReLU(), nn.Linear(in_features=25, out_features=15, bias=True), nn.ReLU())},
    "tcn_median_frequency_with_static_data_fold_2":
            {'optional_model': nn.Sequential(nn.BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU())},
    "tcn_median_frequency_with_static_data_fold_3": {'optional_model': nn.Sequential(nn.ReLU())},
    "tcn_median_frequency_with_static_data_fold_4": {'optional_model': nn.Sequential(nn.ReLU())}
}


def calculate_95_ci(score: Union[float, int], sample_size: int):
    interval = 1.96 * math.sqrt((score * (1 - score)) / sample_size)

    ci = (score - interval, score + interval)

    return ci


def calculate_auc_ap_second_largest_values(df_interval_preds: pd.DataFrame, df_interval_probs: pd.DataFrame,
                                           test_labels, results_dict: Dict):
    """Calculate the AUC, AP F1, precision and recall score for the samples that have one or more
    positive predictions over the num_sub_sequences intervals. The results will be printed.

    The AUC and AP score will be calculated with the mean over the largest two probabilities over
    the num_sub_sequences intervals.

    Parameters
    ----------
    df_interval_preds : pd.Dataframe
        Contains the predictions over all num_sub_sequences intervals for each rec id. Result
        of create_interval_matrix().
    df_interval_probs : pd.DataFrame
        Contains the probabilities over all num_sub_sequences intervals for each rec id. Result
        of create_interval_matrix().
    test_labels :
        True label for each rec id.
    results_dict : Dict
        Dictionary with the results of the evaluated model.
    """
    interval_columns = ['interval_1', 'interval_2', 'interval_3', 'interval_4', 'interval_5']
    boolean_more_than_one_pos_pred = df_interval_preds[interval_columns].sum(axis=1) > 1

    print(f'Precision score with more than 1 pos pred: '
          f'{precision_score(test_labels, boolean_more_than_one_pos_pred.to_list())}')
    print(f'Recall score with more than 1 pos pred: '
          f'{recall_score(test_labels, boolean_more_than_one_pos_pred.to_list())}')
    print(f'F1 score with more than 1 pos pred: {f1_score(test_labels, boolean_more_than_one_pos_pred.to_list())}')

    second_largets_prob = df_interval_probs.loc[:, interval_columns].apply(lambda row: row.nlargest(2).values[-1],
                                                                           axis=1)
    largest_prob = df_interval_probs.loc[:, interval_columns].max(axis=1)
    mean_over_two_largest_probs = pd.concat([second_largets_prob, largest_prob], axis=1).mean(axis=1).values

    print(f'AUC score with mean over largest two probabilities: '
          f'{roc_auc_score(test_labels, mean_over_two_largest_probs)}')
    print(f'AP score with mean over largest two probabilities: '
          f'{average_precision_score(test_labels, mean_over_two_largest_probs)}')

    results_dict.update({'auc_mean_largest_two_probs': roc_auc_score(test_labels, mean_over_two_largest_probs),
                         'ap_mean_largest_two_probs': average_precision_score(test_labels, mean_over_two_largest_probs)})

    return results_dict


def evaluate_tcn_feature_sequence(x_test, x_test_processed, y_test_processed, num_sub_sequences, rec_ids_test_unique,
                                  pos_weight: float, best_params: Dict, features_to_use: List[str],
                                  features_to_use_static: List[str], num_static_features: int, output_path: str,
                                  trained_model_file_name: str):

    test_loader_list, rec_ids_test = generate_dataloader(x_test, x_test_processed, y_test_processed,
                                                         features_to_use, features_to_use_static, rec_ids_test_unique,
                                                         FLAGS.reduced_seq_length, FLAGS.sub_seq_length,
                                                         num_sub_sequences, best_params['batch_size'], test_phase=True)

    n_classes = 1
    # The num_sub_sequences variable is the number of sub_sequences necessary to complete an entire sequence
    num_sub_sequences = int(FLAGS.reduced_seq_length / FLAGS.sub_seq_length)
    device = 'cpu'

    channel_sizes = [best_params['num_hidden_units_per_layer']] * best_params['num_levels']

    if FLAGS.add_static_data and not FLAGS.use_copies_for_static_data:
        input_channels = len(features_to_use)
        input_dim_static = num_static_features
        model_tcn = TCNCombinedModel(input_channels, n_classes, channel_sizes, stride=best_params['stride'],
                                     kernel_size=best_params['kernel_size'], dropout=best_params['drop_out'],
                                     input_dim_static=input_dim_static,
                                     hidden_dim_combined=best_params['num_hidden_units_per_layer'] + best_params['hidden_dim_static'],
                                     model_optional=best_params['optional_model'])

    elif FLAGS.add_static_data and FLAGS.use_copies_for_static_data:
        input_channels = len(features_to_use) + num_static_features
        model_tcn = TCNCombinedModelCopies(input_channels, n_classes, channel_sizes, stride=best_params['stride'],
                                           kernel_size=best_params['kernel_size'], dropout=best_params['drop_out'],
                                           hidden_dim_combined=best_params['num_hidden_units_per_layer'],
                                           model_optional=best_params['optional_model'])

    elif not FLAGS.add_static_data:
        input_channels = len(features_to_use)
        model_tcn = TCN(input_channels, n_classes, channel_sizes, stride=best_params['stride'],
                        kernel_size=best_params['kernel_size'], dropout=best_params['drop_out'])

    model_tcn.to(device)

    num_trainable_params = sum(p.numel() for p in model_tcn.parameters() if p.requires_grad)

    # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
    # https://discuss.pytorch.org/t/bcewithlogitsloss-and-class-weights/88837
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    optimizer = optim.Adam(model_tcn.parameters(), lr=best_params['learning_rate'])

    if FLAGS.add_static_data and not FLAGS.use_copies_for_static_data:
        opt_tcn = OptimizationTCNFeatureSequenceCombined(model=model_tcn, loss_fn=loss_fn, optimizer=optimizer,
                                                         num_sub_sequences=num_sub_sequences, device=device)

    elif FLAGS.add_static_data and FLAGS.use_copies_for_static_data:
        opt_tcn = OptimizationTCNFeatureSequenceCombinedCopies(model=model_tcn, loss_fn=loss_fn, optimizer=optimizer,
                                                               num_sub_sequences=num_sub_sequences, device=device)

    if not FLAGS.add_static_data:
        opt_tcn = OptimizationTCNFeatureSequence(model=model_tcn, loss_fn=loss_fn, optimizer=optimizer,
                                                 num_sub_sequences=num_sub_sequences, device=device)

    ck_point = torch.load(f'{output_path}/{trained_model_file_name}', map_location=torch.device('cpu'))

    if not FLAGS.add_static_data:
        all_test_preds, all_test_probs, test_labels, results_dict = opt_tcn.evaluate(test_loader_list, ck_point)

    if FLAGS.add_static_data:
        all_test_preds, all_test_probs, test_labels, results_dict = opt_tcn.evaluate(test_loader_list,
                                                                                     features_to_use, ck_point)

    df_interval_tcn_preds = create_interval_matrix(all_test_preds, rec_ids_test_unique,
                                                   num_sub_sequences=num_sub_sequences)
    df_interval_tcn_probs = create_interval_matrix(all_test_probs, rec_ids_test_unique,
                                                   num_sub_sequences=num_sub_sequences)

    results_dict = calculate_auc_ap_second_largest_values(df_interval_tcn_preds, df_interval_tcn_probs,
                                                          test_labels, results_dict)

    return all_test_preds, all_test_probs, rec_ids_test, results_dict, df_interval_tcn_probs, num_trainable_params


def evaluate_lstm_feature_sequence(x_test, x_test_processed, y_test_processed, num_sub_sequences, rec_ids_test_unique,
                                   pos_weight: float, best_params: Dict, features_to_use: List[str],
                                   features_to_use_static: List[str], num_static_features: int,
                                   output_path: str, trained_model_file_name: str):

    test_loader_list, rec_ids_test = generate_dataloader(x_test, x_test_processed, y_test_processed,
                                                         features_to_use, features_to_use_static, rec_ids_test_unique,
                                                         FLAGS.reduced_seq_length, FLAGS.sub_seq_length,
                                                         num_sub_sequences, best_params['batch_size'], test_phase=True)

    n_classes = 1
    input_channels = len(features_to_use)
    device = 'cpu'

    if not FLAGS.add_static_data:
        model_lstm = LSTMStatefulClassificationFeatureSequence(input_size=input_channels,
                                                               hidden_size=best_params['hidden_dim'],
                                                               num_layers=best_params['layer_dim'],
                                                               dropout=best_params['drop_out_lstm'],
                                                               output_size=n_classes,
                                                               bidirectional=best_params['bidirectional'],
                                                               batch_size=best_params['batch_size'],
                                                               device=device, batch_first=True)

    elif FLAGS.add_static_data:
        model_lstm = LSTMCombinedModel(input_dim_seq=input_channels, hidden_dim_seq=best_params['hidden_dim_seq'],
                                       input_dim_static=num_static_features,
                                       hidden_dim_static=best_params['hidden_dim_static'],
                                       layer_dim=best_params['layer_dim'], bidirectional=best_params['bidirectional'],
                                       batch_size=best_params['batch_size'], output_dim=n_classes,
                                       model_optional=best_params['optional_model'],
                                       dropout_prob=best_params['drop_out_lstm'], device=device)

    model_lstm.to(device)

    num_trainable_params = sum(p.numel() for p in model_lstm.parameters() if p.requires_grad)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    optimizer = optim.Adam(model_lstm.parameters(), lr=best_params['learning_rate'])

    ck_point = torch.load(f'{output_path}/{trained_model_file_name}', map_location=torch.device('cpu'))

    if not FLAGS.add_static_data:
        opt_model_lstm_stateful_feature = OptimizationStatefulFeatureSequenceLSTM(model=model_lstm,
                                                                                  loss_fn=loss_fn,
                                                                                  optimizer=optimizer,
                                                                                  num_sub_sequences=num_sub_sequences,
                                                                                  device=device)
        all_test_preds, all_test_probs, test_labels, results_dict = opt_model_lstm_stateful_feature.evaluate(
            test_loader_list, ck_point)

    elif FLAGS.add_static_data:
        opt_lstm = OptimizationCombinedLSTM(model=model_lstm, loss_fn=loss_fn, optimizer=optimizer,
                                            num_sub_sequences=num_sub_sequences, device=device)

        all_test_preds, all_test_probs, test_labels, results_dict = opt_lstm.evaluate(test_loader_list,
                                                                                      best_params['optional_model'],
                                                                                      features_to_use, ck_point)

    df_interval_lstm_preds = create_interval_matrix(all_test_preds, rec_ids_test_unique,
                                                    num_sub_sequences=num_sub_sequences)

    df_interval_lstm_probs = create_interval_matrix(all_test_probs, rec_ids_test_unique,
                                                    num_sub_sequences=num_sub_sequences)

    results_dict = calculate_auc_ap_second_largest_values(df_interval_lstm_preds, df_interval_lstm_probs,
                                                          test_labels, results_dict)

    return all_test_preds, all_test_probs, rec_ids_test, results_dict, df_interval_lstm_probs, num_trainable_params


def create_interval_matrix(all_test_results: List, rec_ids_test: List[int], num_sub_sequences: int) -> pd.DataFrame:
    """Create an interval matrix with the predictions/probabilities over each
    sub-sequence (interval) for each rec_id. The resulting df will look like this:

    'interval_1' | 'interval_2' | 'interval_3' | 'interval_4' | 'interval_5' | 'rec_id'.

    all_test_results contains the results (pred/prob) over each sub-sequence of each rec_id in each batch,
    and these results are ordered as follows (in case there are 5 sub-sequences):

    [batch_1_interval_1], [batch_1_interval_2], [batch_1_interval_3], [batch_1_interval_4], [batch_1_interval_5],
    [batch_2_interval_1], [batch_2_interval_2], [batch_2_interval_3], [batch_2_interval_4], [batch_2_interval_5],
    ...
    [batch_n_interval_1], [batch_n_interval_2], [batch_n_interval_3], [batch_n_interval_4], [batch_n_interval_5]

    It is important to mention that the batch sizes do not have to be of equal size. For instance,
    batch 1 can contain 50 samples (and thus have length 50) and batch 2 can contain 10 samples (and
    thus have length 10).

    Parameters
    ----------
    all_test_results : List
        List with all results (preds/probs) on test data. This list is obtained after running the evaluate function
        of lstm/tcn model.
    rec_ids_test : List[int]
        List with the rec ids of the test data, in the EXACT order in which they have been
        passed to the evaluate function of the model.
    num_sub_sequences : int
        Number of sub-sequences in which the original sequence was split into.

    Returns
    -------
    df_interval_matrix : pd.DataFrame
        Dataframe that contains the probs/predictions over each sub-sequence for each rec id.
    """
    # Create a list of all the samples that are present in each batch for each sub-sequence.
    # These batches can have an unequal size. For instance, if there are 60 rec ids in the
    # test set, then the first batch size could be 50 and the second is then 10.
    batch_sizes = [len(samples_in_batch) for all_samples in all_test_results for samples_in_batch in all_samples]

    # This list contains all the results (over all intervals for each rec id) in one long list
    flat_list_all_results = list(np.array(np.concatenate([array for array in all_test_results], axis=1)).flat)

    # We need to wrangle the data in such way that the results over the sub-sequences are linked
    # to the correct rec id. The reason we need to do this data wrangling is because the batch sizes
    # can be of unequal size and this makes the process a bit less straightforward
    cumsum_batch_sizes = list(np.cumsum(batch_sizes))
    # Insert value 0 at the beginning of the cumulative list of samples in the batches
    cumsum_batch_sizes.insert(0, 0)
    # Remove the last entry of the list
    cumsum_batch_sizes = cumsum_batch_sizes[:-1]

    # This list contains the same results as all_test_results but now stored in a more convenient way
    list_of_lists_with_results = [flat_list_all_results[cumsum_batch_size:cumsum_batch_size + batch_size] for
                                  cumsum_batch_size, batch_size in zip(cumsum_batch_sizes, batch_sizes)]

    # Here we make the actual interval matrix
    df_interval_matrix = pd.DataFrame()
    for i in list(range(0, len(list_of_lists_with_results), num_sub_sequences)):
        df_batch_results = pd.DataFrame.from_records(list_of_lists_with_results[i:(i + num_sub_sequences)]).transpose()
        df_interval_matrix = pd.concat([df_interval_matrix, df_batch_results], axis=0)
    df_interval_matrix = df_interval_matrix.reset_index(drop=True)

    df_interval_matrix = pd.concat([df_interval_matrix, pd.DataFrame(rec_ids_test)], axis=1)
    df_interval_matrix.columns = ['interval_1', 'interval_2', 'interval_3', 'interval_4', 'interval_5', 'rec_id']

    return df_interval_matrix


def cross_validation_evaluation_baseline_model():
    df_clinical_information = build_clinical_information_dataframe(data_path, settings_path)
    df_static_information = basic_preprocessing_static_data(data_path, settings_path, df_clinical_information)

    auc_scores = []
    ap_scores = []

    for outer_fold_i in range(FLAGS.n_folds):
        _, rec_ids_train_outer_fold, rec_ids_test_outer_fold = get_best_params_comb_model(FLAGS.hyperoptimization_file_name,
                                                                                          outer_fold_i=outer_fold_i)

        print(f'fold: {outer_fold_i}')

        X_train_static_fold = df_static_information.loc[df_static_information[c.REC_ID_NAME].
            isin(rec_ids_train_outer_fold)].copy().reset_index(drop=True)
        X_test_static_fold = df_static_information.loc[df_static_information[c.REC_ID_NAME].
            isin(rec_ids_test_outer_fold)].copy().reset_index(drop=True)

        pos_cases = X_train_static_fold['premature'].value_counts()[1]
        neg_cases = X_train_static_fold['premature'].value_counts()[0]
        pos_weight = neg_cases / pos_cases
        print(f'pos weight: {pos_weight}')

        x_arr_static_train, y_arr_static_train, selected_columns_train_static, rec_id_list_static_train = preprocess_static_data(
            X_train_static_fold,
            X_train_static_fold,
            threshold_correlation=0.85)

        num_static_features = len(selected_columns_train_static)
        print(f'Num static features: {num_static_features}')

        x_arr_static_test, y_arr_static_test, selected_columns_test_static, rec_id_list_static_test = preprocess_static_data(
            X_train_static_fold,
            X_test_static_fold,
            threshold_correlation=0.85)

        logreg = LogisticRegression(penalty='l2', C=1, class_weight={0: 1, 1: pos_weight})
        logreg.fit(x_arr_static_train, y_arr_static_train)

        print(
            f'Average precision score: {average_precision_score(y_arr_static_test, logreg.predict_proba(x_arr_static_test)[:, 1])}')
        print(f'AUC score: {roc_auc_score(y_arr_static_test, logreg.predict_proba(x_arr_static_test)[:, 1])}')

        ap_score_fold = average_precision_score(y_arr_static_test, logreg.predict_proba(x_arr_static_test)[:, 1])
        auc_score_fold = roc_auc_score(y_arr_static_test, logreg.predict_proba(x_arr_static_test)[:, 1])

        ap_scores.append(ap_score_fold)
        auc_scores.append(auc_score_fold)

    print(f'Mean AP score: {np.mean(ap_scores)}')
    print(f'Std AP score: {np.std(ap_scores)}')

    print(f'Mean AUC score: {np.mean(auc_scores)}')
    print(f'Std AUC score: {np.std(auc_scores)}')


def cross_validation_evaluation(model_name):
    # Load dataset from hard disk
    # Original signal data
    df_signals_new = pd.read_csv(f'{data_path}/df_signals_filt.csv', sep=';')

    df_clinical_information = build_clinical_information_dataframe(data_path, settings_path)

    df_features, df_label = basic_preprocessing_signal_data(df_signals_new, df_clinical_information,
                                                            FLAGS.reduced_seq_length, features_to_use,
                                                            FLAGS.feature_name, fs)

    df_total = pd.concat([df_features, df_label], axis=1)

    # The number of sub-sequences needed to make up an original sequence
    num_sub_sequences_fixed = int(FLAGS.reduced_seq_length / FLAGS.sub_seq_length)

    if FLAGS.add_static_data:
        df_static_information = basic_preprocessing_static_data(data_path, settings_path, df_clinical_information)

    current_date_and_time = "{:%Y-%m-%d_%H-%M}".format(datetime.datetime.now())

    df_interval_probs = pd.DataFrame()
    num_trainable_params_list = []

    for outer_fold_i in range(FLAGS.n_folds):
        if FLAGS.add_static_data:
            best_params_model_name_fold = f'{FLAGS.model}_{FLAGS.feature_name}_with_static_data_fold_{outer_fold_i}'

            _, rec_ids_train_outer_fold, rec_ids_test_outer_fold = get_best_params_comb_model(FLAGS.hyperoptimization_file_name,
                                                                                              outer_fold_i=outer_fold_i)

        elif not FLAGS.add_static_data:
            best_params_model_name_fold = f'{FLAGS.model}_{FLAGS.feature_name}_fold_{outer_fold_i}'

            _, rec_ids_train_outer_fold, rec_ids_test_outer_fold = get_best_params(FLAGS.hyperoptimization_file_name,
                                                                                   outer_fold_i=outer_fold_i)

        best_params: dict = read_settings(settings_path, 'best_params')
        best_params = best_params[best_params_model_name_fold]

        # If static data is added to the model, add the optional model part to the best_params dict
        if FLAGS.add_static_data:
            best_params.update(optional_model_dict[best_params_model_name_fold])

        print(best_params)
        print(f'fold: {outer_fold_i}')

        final_model: dict = read_settings(settings_path, 'final_models')
        trained_model_file_name = final_model[best_params_model_name_fold]

        print(f'model name: {trained_model_file_name}')

        df_train_outer_fold = df_total.loc[df_total[c.REC_ID_NAME].isin(rec_ids_train_outer_fold)].copy().reset_index(drop=True)
        df_test_outer_fold = df_total.loc[df_total[c.REC_ID_NAME].isin(rec_ids_test_outer_fold)].copy().reset_index(drop=True)

        X_train_signal_fold, y_train_fold = feature_label_split(df_train_outer_fold, 'premature')
        X_test_signal_fold, y_test_fold = feature_label_split(df_test_outer_fold, 'premature')

        pos_cases = y_train_fold['premature'].value_counts()[1]
        neg_cases = y_train_fold['premature'].value_counts()[0]
        pos_weight = neg_cases / pos_cases
        print(f'pos weight: {pos_weight}')

        # We keep the rec ids order to later on merge the static data correctly
        rec_ids_x_train_signal = list(X_train_signal_fold[c.REC_ID_NAME])
        rec_ids_x_test_signal = list(X_test_signal_fold[c.REC_ID_NAME])

        X_train_signal_fold_processed, X_test_signal_fold_processed, y_train_fold_processed, y_test_fold_processed = \
            preprocess_signal_data(X_train_signal_fold, X_test_signal_fold, y_train_fold, y_test_fold, features_to_use)

        if FLAGS.add_static_data:
            X_train_static_fold = df_static_information.loc[df_static_information[c.REC_ID_NAME].
                isin(rec_ids_train_outer_fold)].copy().reset_index(drop=True)
            X_test_static_fold = df_static_information.loc[df_static_information[c.REC_ID_NAME].
                isin(rec_ids_test_outer_fold)].copy().reset_index(drop=True)

            assert all(x == y for x, y in zip(X_train_signal_fold[c.REC_ID_NAME].unique(),
                                              X_train_static_fold[c.REC_ID_NAME].unique())), \
                "Rec ids in X_train_signal and X_train_static must be in the exact same order!"

            assert all(x == y for x, y in zip(X_test_signal_fold[c.REC_ID_NAME].unique(),
                                              X_test_static_fold[c.REC_ID_NAME].unique())), \
                "Rec ids in X_test_signal and X_test_static must be in the exact same order!"

            assert set(X_train_signal_fold[c.REC_ID_NAME]) == set(X_train_static_fold[c.REC_ID_NAME]), \
                "Rec ids in train signals and train static data are not the same!"

            assert set(X_test_signal_fold[c.REC_ID_NAME]) == set(X_test_static_fold[c.REC_ID_NAME]), \
                "Rec ids in test signals and test static data are not the same!"

            X_train_combined_fold, X_test_combined_fold, X_train_combined_processed, X_test_combined_processed, \
            selected_columns_train_static = add_static_data_to_signal_data(X_train_static_fold, X_test_static_fold,
                                                                           X_train_signal_fold, X_test_signal_fold,
                                                                           X_train_signal_fold_processed,
                                                                           X_test_signal_fold_processed,
                                                                           rec_ids_x_train_signal,
                                                                           rec_ids_x_test_signal, features_to_use,
                                                                           threshold_correlation=0.85)

        if model_name == 'tcn' and not FLAGS.add_static_data:
            features_to_use_static = []

            all_test_preds, all_test_probs, rec_ids_test, results_dict, df_interval, \
            num_trainable_params = evaluate_tcn_feature_sequence(X_test_signal_fold, X_test_signal_fold_processed,
                                                                 y_test_fold_processed, num_sub_sequences_fixed,
                                                                 rec_ids_test_outer_fold, pos_weight, best_params,
                                                                 features_to_use, features_to_use_static,
                                                                 num_static_features=len(features_to_use_static),
                                                                 output_path=out_path_model,
                                                                 trained_model_file_name=trained_model_file_name)

        elif model_name == 'tcn' and FLAGS.add_static_data:
            all_test_preds, all_test_probs, rec_ids_test, results_dict, df_interval, \
            num_trainable_params = evaluate_tcn_feature_sequence(X_test_combined_fold, X_test_combined_processed,
                                                                 y_test_fold_processed, num_sub_sequences_fixed,
                                                                 rec_ids_test_outer_fold, pos_weight, best_params,
                                                                 features_to_use, selected_columns_train_static,
                                                                 num_static_features=len(selected_columns_train_static),
                                                                 output_path=out_path_model,
                                                                 trained_model_file_name=trained_model_file_name)

        elif model_name == 'lstm' and not FLAGS.add_static_data:
            features_to_use_static = []

            all_test_preds, all_test_probs, rec_ids_test, results_dict, df_interval, \
            num_trainable_params = evaluate_lstm_feature_sequence(X_test_signal_fold, X_test_signal_fold_processed,
                                                                  y_test_fold_processed, num_sub_sequences_fixed,
                                                                  rec_ids_test_outer_fold, pos_weight, best_params,
                                                                  features_to_use, features_to_use_static,
                                                                  num_static_features=len(features_to_use_static),
                                                                  output_path=out_path_model,
                                                                  trained_model_file_name=trained_model_file_name)

        elif model_name == 'lstm' and FLAGS.add_static_data:
            all_test_preds, all_test_probs, rec_ids_test, results_dict, df_interval, \
            num_trainable_params = evaluate_lstm_feature_sequence(X_test_combined_fold, X_test_combined_processed,
                                                                  y_test_fold_processed, num_sub_sequences_fixed,
                                                                  rec_ids_test_outer_fold, pos_weight, best_params,
                                                                  features_to_use, selected_columns_train_static,
                                                                  num_static_features=len(selected_columns_train_static),
                                                                  output_path=out_path_model,
                                                                  trained_model_file_name=trained_model_file_name)

        results_dict.update({'model_file_name': trained_model_file_name})
        print(results_dict)

        with open(f'{evaluation_results_path}/{best_params_model_name_fold}_results_{current_date_and_time}.csv', 'w') as f:
            for key in results_dict.keys():
                f.write("%s, %s\n" % (key, results_dict[key]))

        print(f'Results are saved at: {evaluation_results_path}/{best_params_model_name_fold}_results_{current_date_and_time}.csv')
        print(f'All test probs: {all_test_probs}')

        num_trainable_params_list.append(num_trainable_params)

        # Add outer fold as column
        df_interval = pd.concat([df_interval, pd.DataFrame([outer_fold_i]*len(rec_ids_test_outer_fold))], axis=1)

        df_interval_probs = pd.concat([df_interval_probs, df_interval], axis=0)
    df_interval_probs.to_csv(f'{evaluation_results_path}/{best_params_model_name}_interval_predictions.csv')

    print(f'Min number of trainable params over all folds: {np.min(num_trainable_params_list)}')
    print(f'Max number of trainable params over all folds: {np.max(num_trainable_params_list)}')

    auc_mean_prob_list = []
    ap_mean_prob_list = []
    auc_max_prob_list = []
    ap_max_prob_list = []
    for i in range(FLAGS.n_folds):
        path_to_results = f'{evaluation_results_path}/{best_params_model_name}_fold_{i}_results_{current_date_and_time}.csv'

        df_result = pd.read_csv(f'{path_to_results}', header=None)
        df_result = df_result.rename(columns={0: 'metric', 1: 'score'})
        auc_mean_prob = float(df_result.loc[df_result['metric'] == 'auc_mean_prob']['score'].values)
        ap_mean_prob = float(df_result.loc[df_result['metric'] == 'ap_mean_prob']['score'].values)

        auc_max_prob = float(df_result.loc[df_result['metric'] == 'auc_max_prob']['score'].values)
        ap_max_prob = float(df_result.loc[df_result['metric'] == 'ap_max_prob']['score'].values)

        auc_mean_prob_list.append(auc_mean_prob)
        ap_mean_prob_list.append(ap_mean_prob)
        auc_max_prob_list.append(auc_max_prob)
        ap_max_prob_list.append(ap_max_prob)

    print(f'AUC mean prob over all folds: {np.mean(auc_mean_prob_list)}')
    print(f'AUC mean prob std over all folds: {np.std(auc_mean_prob_list)}')

    print(f'AP mean prob over all folds: {np.mean(ap_mean_prob_list)}')
    print(f'AP mean prob std over all folds: {np.std(ap_mean_prob_list)}')

    print(f'AUC max prob over all folds: {np.mean(auc_max_prob_list)}')
    print(f'AUC max prob std over all folds: {np.std(auc_max_prob_list)}')

    print(f'AP max prob over all folds: {np.mean(ap_max_prob_list)}')
    print(f'AP max prob std over all folds: {np.std(ap_max_prob_list)}')


def main(trained_model_file_name: Union[str, Dict], features_to_use: List[str], best_params: Dict, output_path: str):
    df_signals_new = pd.read_csv(f'{data_path}/df_signals_filt.csv', sep=';')

    df_clinical_information = build_clinical_information_dataframe(data_path, settings_path)
    df_signals = df_signals_new.merge(df_clinical_information[[c.REC_ID_NAME, 'premature']],
                                      how='left', on=c.REC_ID_NAME)

    df_demographics = build_demographics_dataframe(data_path, settings_path)
    df_static_information = df_demographics.merge(df_clinical_information, how='left', on=c.REC_ID_NAME)

    # Assign 0 to non-premature cases and assign 1 to premature cases
    lb = LabelEncoder()
    df_signals['premature'] = lb.fit_transform(df_signals['premature'])

    X_train, X_val, X_test, _, _, _ = train_val_test_split(df_clinical_information, 'premature', test_ratio=0.2,
                                                           shuffle=True, random_state=0)

    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)

    # Count the number of static features
    if FLAGS.add_static_data:
        _, _, selected_columns_fit_static, _, _ = preprocess_static_data(df_static_information, X_train, X_train,
                                                                         threshold_correlation=0.85)

        num_static_features = len(selected_columns_fit_static)

    if not FLAGS.add_static_data:
        num_static_features = 0

    if FLAGS.model == 'tcn':
        all_test_preds, all_test_probs, rec_ids_test, results_dict = evaluate_tcn_feature_sequence(df_signals,
                                                                                                   df_clinical_information,
                                                                                                   df_static_information,
                                                                                                   X_train_val, X_test,
                                                                                                   trained_model_file_name,
                                                                                                   features_to_use,
                                                                                                   best_params,
                                                                                                   add_static_data=FLAGS.add_static_data,
                                                                                                   copies=FLAGS.use_copies_for_static_data,
                                                                                                   num_static_features=num_static_features,
                                                                                                   output_path=output_path)

    elif FLAGS.model == 'lstm':
        all_test_preds, all_test_probs, rec_ids_test, results_dict = evaluate_lstm_feature_sequence(df_signals,
                                                                                                    df_clinical_information,
                                                                                                    df_static_information,
                                                                                                    X_train_val, X_test,
                                                                                                    trained_model_file_name,
                                                                                                    features_to_use,
                                                                                                    best_params,
                                                                                                    add_static_data=FLAGS.add_static_data,
                                                                                                    num_static_features=num_static_features,
                                                                                                    output_path=output_path)

    return all_test_preds, all_test_probs, rec_ids_test, results_dict


if __name__ == "__main__":
    out_path_model = os.path.abspath("trained_models")

    if not os.path.isdir(out_path_model):
        os.mkdir(out_path_model)

    evaluation_results_path = os.path.join(file_paths['output_path'], 'evaluation_results')

    if not os.path.isdir(evaluation_results_path):
        os.mkdir(evaluation_results_path)

    # Command line arguments
    parser = argparse.ArgumentParser(description='Evaluate the final model on test set using the final trained model'
                                                 'that is obtained after running final_train.py. The performance is'
                                                 'given in terms of AUC and AP. The optimal hyperparameters have to be '
                                                 'put in the best_params.json file and the optional_model part has to '
                                                 'be put in the optional_model_dict and bidirectional_lstm_dict, which '
                                                 'are placed at the top of this evaluation.py file. The name of your '
                                                 'final model has to be specified in the final_models.json file and at '
                                                 'the top of this main function you can specify in which folder your '
                                                 'final model is saved.')

    parser.add_argument('--model', type=str, required=True,
                        help="Select what model to use: 'lstm' or 'tcn'",
                        choices=['tcn', 'lstm'])

    parser.add_argument('--feature_name', type=str, required=True,
                        help="Select what feature to use for data reduction: 'sample_entropy', 'peak_frequency' or "
                             "'median_frequency'",
                        choices=['sample_entropy', 'peak_frequency', 'median_frequency'])

    parser.add_argument('--hyperoptimization_file_name', type=str, required=True)

    parser.add_argument('--n_folds', type=int, required=True)

    parser.add_argument('--reduced_seq_length', type=int, required=True, default=50,
                        help="The time window length of which you want to calculate feature_name on each time step."
                             "For example, if reduced_seq_length is 50 and feature_name is sample entropy, then you'll "
                             "end up with 50 values of the sample entropy which are calculated over non-overlapping "
                             "time windows from df_signals_new.")
    parser.add_argument('--sub_seq_length', type=int, required=True, default=10,
                        help="The number of time steps you want to use to split reduced_seq_length into. For example, "
                             "if reduced_seq_length is 50 and sub_seq_length is 10, then you'll have 5 sub-sequences "
                             "that make up the total reduced_seq_length. A prediction will be made over each "
                             "sub-sequence")

    # Make a dependency such that it is required to have either the --add_static_data or the --no_static_data flag
    parser.add_argument('--add_static_data', action='store_true',
                        required=('--model' in sys.argv and '--no_static_data' not in sys.argv),
                        help="Add static clinical data to the model. Use either the --add_static_data or the "
                             "--no_static_data flag")
    parser.add_argument('--no_static_data', dest='add_static_data', action='store_false',
                        required=('--model' in sys.argv and '--add_static_data' not in sys.argv),
                        help="Use only the EHG data for modeling. Use either the --add_static_data or the "
                             "--no_static_data flag")
    parser.set_defaults(add_static_data=True)

    # Make a dependency such that it is required to have either the --use_copies_for_static_data or the
    # --no_copies_for_static_data flag if the --add_static_data flag is present
    parser.add_argument('--use_copies_for_static_data', action='store_true',
                        required=('--add_static_data' in sys.argv and '--no_copies_for_static_data' not in sys.argv),
                        help="The static data is now treated as a time series, were each (static) value of each "
                             "variable is copied along the time steps of the EHG time series data." 
                             "Meaning, if there are 10 time steps in the seq data, then the static data is also "
                             "copied for 10 time steps. This flag or the --no_copies_for_static_data flag are only "
                             "required if the --add_static_data flag is used.")
    parser.add_argument('--no_copies_for_static_data', dest='use_copies_for_static_data', action='store_false',
                        required=('--add_static_data' in sys.argv and '--use_copies_for_static_data' not in sys.argv),
                        help="The static data is now treated as single values that will be concatenated separately to "
                             "the time series data after the time series data has been processed. Use either the "
                             "--use_copies_for_static_data or the --no_copies_for_static_data flag. This flag or the "
                             "--use_copies_for_static_data flag are only required if the --add_static_data flag "
                             "is used.")
    parser.set_defaults(use_copies_for_static_data=False)

    # Make a dependency such that it is required to have either the --baseline or the --no_baseline flag
    parser.add_argument('--baseline', action='store_true',
                        required=('--no_baseline' not in sys.argv),
                        help="Calculate performance of logistic regression baseline on static data")
    parser.add_argument('--no_baseline', dest='baseline', action='store_false',
                        required=('--baseline' not in sys.argv),
                        help="Run normal evaluation.")
    parser.set_defaults(add_static_data=True)

    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

    if FLAGS.add_static_data:
        best_params_model_name = f'{FLAGS.model}_{FLAGS.feature_name}_with_static_data'

    elif not FLAGS.add_static_data:
        best_params_model_name = f'{FLAGS.model}_{FLAGS.feature_name}'

    features_to_use = ['channel_1_filt_0.34_1_hz', 'channel_2_filt_0.34_1_hz', 'channel_3_filt_0.34_1_hz']
    fs = 20

    if FLAGS.baseline:
        cross_validation_evaluation_baseline_model()

    else:
        cross_validation_evaluation(FLAGS.model)
