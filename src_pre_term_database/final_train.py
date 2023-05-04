import pandas as pd
from src_pre_term_database.optimization import OptimizationTCNFeatureSequence, \
    OptimizationStatefulFeatureSequenceLSTM, OptimizationCombinedLSTM, OptimizationTCNFeatureSequenceCombined, \
    OptimizationTCNFeatureSequenceCombinedCopies
from src_pre_term_database.modeling import TCN, LSTMStatefulClassificationFeatureSequence, \
    LSTMCombinedModel, TCNCombinedModel, TCNCombinedModelCopies
from src_pre_term_database.load_dataset import build_clinical_information_dataframe
from src_pre_term_database.data_processing_and_feature_engineering import preprocess_signal_data, \
    basic_preprocessing_signal_data, one_hot_encode_static_data, add_static_data_to_signal_data, generate_dataloader, \
    feature_label_split
from src_pre_term_database.utils import read_settings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict
import constants as c
import argparse
import sys
import os
import json
import ast

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
        {'optional_model': nn.Sequential(nn.BatchNorm1d(22, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU())},
    "lstm_peak_frequency_with_static_data_fold_1":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(31, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU())},
    "lstm_peak_frequency_with_static_data_fold_2":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU())},
    "lstm_peak_frequency_with_static_data_fold_3":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(35, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU())},
    "lstm_peak_frequency_with_static_data_fold_4":
        {'optional_model': nn.Sequential(nn.ReLU(), nn.Linear(in_features=28, out_features=17, bias=True), nn.ReLU())},


    "lstm_median_frequency_with_static_data_fold_0":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True), nn.ReLU())},
    "lstm_median_frequency_with_static_data_fold_1":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU())},
    "lstm_median_frequency_with_static_data_fold_2":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU())},
    "lstm_median_frequency_with_static_data_fold_3":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU())},
    "lstm_median_frequency_with_static_data_fold_4":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(31, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=31, out_features=20, bias=True), nn.ReLU())},

    "tcn_sample_entropy_with_static_data_fold_0":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(35, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=35, out_features=16, bias=True), nn.ReLU())},
    "tcn_sample_entropy_with_static_data_fold_1":
        {'optional_model': nn.Sequential(nn.ReLU(), nn.Linear(in_features=31, out_features=15, bias=True), nn.ReLU())},
    "tcn_sample_entropy_with_static_data_fold_2":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(28, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=28, out_features=13, bias=True), nn.ReLU())},
    "tcn_sample_entropy_with_static_data_fold_3":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(28, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=28, out_features=18, bias=True), nn.ReLU())},
    "tcn_sample_entropy_with_static_data_fold_4":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(31, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
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
        {'optional_model': nn.Sequential(nn.BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=29, out_features=14, bias=True), nn.ReLU())},
    "tcn_median_frequency_with_static_data_fold_0": {'optional_model': nn.Sequential(nn.ReLU())},
    "tcn_median_frequency_with_static_data_fold_1":
        {'optional_model': nn.Sequential(nn.ReLU(), nn.Linear(in_features=25, out_features=15, bias=True), nn.ReLU())},
    "tcn_median_frequency_with_static_data_fold_2":
            {'optional_model': nn.Sequential(nn.BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True,
                                                            track_running_stats=True), nn.ReLU())},
    "tcn_median_frequency_with_static_data_fold_3": {'optional_model': nn.Sequential(nn.ReLU())},
    "tcn_median_frequency_with_static_data_fold_4": {'optional_model': nn.Sequential(nn.ReLU())}
}


def final_train_lstm_feature_sequence(x_train, x_train_processed, y_train_processed, num_sub_sequences: int,
                                      rec_ids_train: List[int], pos_weight: np.float, best_params: Dict,
                                      features_to_use: List[str], features_to_use_static: List[str],
                                      num_static_features: int, fold_i: int):

    train_loader_list = generate_dataloader(x_train, x_train_processed, y_train_processed,
                                            features_to_use, features_to_use_static, rec_ids_train,
                                            FLAGS.reduced_seq_length, FLAGS.sub_seq_length, num_sub_sequences,
                                            best_params['batch_size'], test_phase=False)

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

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    optimizer = optim.Adam(model_lstm.parameters(), lr=best_params['learning_rate'])

    if not FLAGS.add_static_data:
        opt_model_lstm_stateful_feature = OptimizationStatefulFeatureSequenceLSTM(model=model_lstm,
                                                                                  loss_fn=loss_fn,
                                                                                  optimizer=optimizer,
                                                                                  num_sub_sequences=num_sub_sequences,
                                                                                  device=device)

        opt_model_lstm_stateful_feature.final_train(train_loader_list,
                                                    feature_name=best_params['feature_name'],
                                                    add_static_data=FLAGS.add_static_data,
                                                    fold_i=fold_i,
                                                    n_epochs=best_params['num_epochs'])

    if FLAGS.add_static_data:
        opt_lstm = OptimizationCombinedLSTM(model=model_lstm, loss_fn=loss_fn, optimizer=optimizer,
                                            num_sub_sequences=num_sub_sequences, device=device)

        opt_lstm.final_train(train_loader_list, feature_name=best_params['feature_name'],
                             add_static_data=FLAGS.add_static_data, fold_i=fold_i, features_to_use=features_to_use,
                             model_optional=best_params['optional_model'], n_epochs=best_params['num_epochs'])


def final_train_tcn_feature_sequence(x_train, x_train_processed, y_train_processed, num_sub_sequences: int,
                                     rec_ids_train: List[int], pos_weight: np.float, best_params: Dict,
                                     features_to_use: List[str], features_to_use_static: List[str],
                                     num_static_features: int, fold_i: int):

    train_loader_list = generate_dataloader(x_train, x_train_processed, y_train_processed,
                                            features_to_use, features_to_use_static, rec_ids_train,
                                            FLAGS.reduced_seq_length, FLAGS.sub_seq_length, num_sub_sequences,
                                            best_params['batch_size'], test_phase=False)

    n_classes = 1
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

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    optimizer = optim.Adam(model_tcn.parameters(), lr=best_params['learning_rate'])

    if FLAGS.add_static_data and not FLAGS.use_copies_for_static_data:
        opt_tcn = OptimizationTCNFeatureSequenceCombined(model=model_tcn, loss_fn=loss_fn, optimizer=optimizer,
                                                         num_sub_sequences=num_sub_sequences, device=device)
        opt_tcn.final_train(train_loader_list, features_to_use, feature_name=best_params['feature_name'],
                            add_static_data=FLAGS.add_static_data, fold_i=fold_i, n_epochs=best_params['num_epochs'])

    elif FLAGS.add_static_data and FLAGS.use_copies_for_static_data:
        opt_tcn = OptimizationTCNFeatureSequenceCombinedCopies(model=model_tcn, loss_fn=loss_fn, optimizer=optimizer,
                                                               num_sub_sequences=num_sub_sequences, device=device)

        opt_tcn.final_train(train_loader_list, feature_name=best_params['feature_name'],
                            add_static_data=FLAGS.add_static_data, fold_i=fold_i, n_epochs=best_params['num_epochs'])

    elif not FLAGS.add_static_data:
        opt_tcn = OptimizationTCNFeatureSequence(model=model_tcn, loss_fn=loss_fn, optimizer=optimizer,
                                                 num_sub_sequences=num_sub_sequences, device=device)

        opt_tcn.final_train(train_loader_list, feature_name=best_params['feature_name'],
                            add_static_data=FLAGS.add_static_data, fold_i=fold_i, n_epochs=best_params['num_epochs'])


def cross_validation_final_train(model_name: str):
    """Final train the cross validation models. For each fold, the corresponding train data will be selected
    and a model with the optimal hyperparameter settings will be trained on that specific fold.

    Parameters
    ----------
    model_name : str
        Name of the model which you want to final train. Either 'lstm' or 'tcn'.
    """
    # Load dataset from hard disk
    # Original signal data
    df_signals_new = pd.read_csv(f'{data_path}/df_signals_filt.csv', sep=';')

    df_clinical_information = build_clinical_information_dataframe(data_path, settings_path)

    df_features, df_label = basic_preprocessing_signal_data(df_signals_new, df_clinical_information,
                                                            FLAGS.reduced_seq_length, features_to_use,
                                                            FLAGS.feature_name, fs)

    df_total = pd.concat([df_features, df_label], axis=1)
    df_ground_truth = pd.concat([df_features[c.REC_ID_NAME], df_label], axis=1)

    # The number of sub-sequences needed to make up an original sequence
    num_sub_sequences_fixed = int(FLAGS.reduced_seq_length / FLAGS.sub_seq_length)

    path_to_optimal_params = f'{hyperopt_path}/{FLAGS.hyperoptimization_file_name}'

    if FLAGS.add_static_data:
        df_static_information = one_hot_encode_static_data(data_path, settings_path, df_clinical_information)

    for outer_fold_i in range(FLAGS.n_folds):

        if FLAGS.add_static_data:
            best_params_model_name = f'{FLAGS.model}_{FLAGS.feature_name}_with_static_data_fold_{outer_fold_i}'

            best_params: dict = read_settings(settings_path, 'best_params')
            best_params = best_params[best_params_model_name]
            # If static data is added to the model, add the optional model part to the best_params dict
            best_params.update(optional_model_dict[best_params_model_name])

            _, rec_ids_train_outer_fold, rec_ids_test_outer_fold = get_best_params(path_to_optimal_params,
                                                                                   outer_fold_i=outer_fold_i,
                                                                                   add_static_data=FLAGS.add_static_data)

        elif not FLAGS.add_static_data:
            best_params_model, rec_ids_train_outer_fold, rec_ids_test_outer_fold = get_best_params(path_to_optimal_params,
                                                                                                   outer_fold_i=outer_fold_i,
                                                                                                   add_static_data=FLAGS.add_static_data)

            best_params_model_name = f'{FLAGS.model}_{FLAGS.feature_name}_fold_{outer_fold_i}'

            best_params_file: dict = read_settings(settings_path, 'best_params')

            best_params_file[best_params_model_name] = best_params_model

            complete_filename = os.path.join(settings_path, 'best_params' + '.json')

            with open(complete_filename, 'w') as outfile:
                json.dump(best_params_file, outfile)

            best_params = best_params_file[best_params_model_name]

        df_train_outer_fold = df_total.loc[df_total[c.REC_ID_NAME].isin(rec_ids_train_outer_fold)].copy().reset_index(drop=True)
        df_test_outer_fold = df_total.loc[df_total[c.REC_ID_NAME].isin(rec_ids_test_outer_fold)].copy().reset_index(drop=True)

        X_train_signal_fold, y_train_fold = feature_label_split(df_train_outer_fold, 'premature')
        X_test_signal_fold, y_test_fold = feature_label_split(df_test_outer_fold, 'premature')

        pos_cases = y_train_fold['premature'].value_counts()[1]
        neg_cases = y_train_fold['premature'].value_counts()[0]

        pos_weight = neg_cases / pos_cases

        # We keep the rec ids order to later on merge the static data correctly
        rec_ids_x_train_signal = list(X_train_signal_fold[c.REC_ID_NAME])
        rec_ids_x_test_signal = list(X_test_signal_fold[c.REC_ID_NAME])

        X_train_signal_fold_processed, X_test_signal_fold_processed, y_train_fold_processed, y_test_fold_processed = \
            preprocess_signal_data(X_train_signal_fold, X_test_signal_fold, y_train_fold, y_test_fold, features_to_use)

        label_check_train = df_ground_truth.loc[df_ground_truth[c.REC_ID_NAME].isin(rec_ids_train_outer_fold), 'premature'].copy().reset_index(drop=True)
        label_check_test = df_ground_truth.loc[df_ground_truth[c.REC_ID_NAME].isin(rec_ids_test_outer_fold), 'premature'].copy().reset_index(drop=True)

        assert all(x == y for x, y in zip(y_train_fold_processed, label_check_train.values))
        assert all(x == y for x, y in zip(y_test_fold_processed, label_check_test.values))

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
            final_train_tcn_feature_sequence(X_train_signal_fold, X_train_signal_fold_processed, y_train_fold_processed,
                                             num_sub_sequences_fixed, rec_ids_train_outer_fold, pos_weight, best_params,
                                             features_to_use, features_to_use_static,
                                             num_static_features=len(features_to_use_static), fold_i=outer_fold_i)

        elif model_name == 'tcn' and FLAGS.add_static_data:
            final_train_tcn_feature_sequence(X_train_combined_fold, X_train_combined_processed, y_train_fold_processed,
                                             num_sub_sequences_fixed, rec_ids_train_outer_fold, pos_weight, best_params,
                                             features_to_use, selected_columns_train_static,
                                             num_static_features=len(selected_columns_train_static),
                                             fold_i=outer_fold_i)

        elif model_name == 'lstm' and not FLAGS.add_static_data:
            features_to_use_static = []
            final_train_lstm_feature_sequence(X_train_signal_fold, X_train_signal_fold_processed,
                                              y_train_fold_processed, num_sub_sequences_fixed, rec_ids_train_outer_fold,
                                              pos_weight, best_params, features_to_use, features_to_use_static,
                                              num_static_features=len(features_to_use_static), fold_i=outer_fold_i)

        elif model_name == 'lstm' and FLAGS.add_static_data:
            final_train_lstm_feature_sequence(X_train_combined_fold, X_train_combined_processed, y_train_fold_processed,
                                              num_sub_sequences_fixed, rec_ids_train_outer_fold, pos_weight,
                                              best_params, features_to_use, selected_columns_train_static,
                                              num_static_features=len(selected_columns_train_static),
                                              fold_i=outer_fold_i)


def get_best_params(path_to_optimal_params: str, outer_fold_i: int, add_static_data: bool):
    """Get the best hyperparameters for outer_fold_i.

    Parameters
    ----------
    path_to_optimal_params : str
        Path of where the optimal hyperparameters are stored.
    outer_fold_i : int
        Number of the fold.
    add_static_data : bool
        True or False. Whether you want to have the model on only EHG or both EHG + static data.

    Returns
    ----------
    best_params : Dict
        Dictionary containing the best hyperparameters
    rec_ids_train_outer_fold: List[int]
        List with the rec_ids that were used in the train fold.
    rec_ids_test_outer_fold: List[int]
        List with the rec_ids that were used in the test fold.
    """
    hyper_opt_params = pd.read_csv(f'{path_to_optimal_params}')

    # Take the mean auc over the mean/max probabilities of the sub-sequences over all outer folds
    df_mean_auc_per_outer_fold = hyper_opt_params.groupby(['params', 'outer_fold']).agg({'mean_prob_auc': np.mean,
                                                                                         'max_prob_auc': np.mean,
                                                                                         'rec_ids_train_outer': lambda x: x.unique(),
                                                                                         'rec_ids_test_outer': lambda x: x.unique(),
                                                                                         'rec_ids_train_inner': lambda x: x.unique(),
                                                                                         'rec_ids_test_inner': lambda x: x.unique()}).reset_index()

    columns_to_change = ['rec_ids_train_outer', 'rec_ids_test_outer', 'rec_ids_train_inner', 'rec_ids_test_inner']

    for column in columns_to_change:

        if column in ['rec_ids_train_outer', 'rec_ids_test_outer']:
            df_mean_auc_per_outer_fold[column] = df_mean_auc_per_outer_fold[column].map(lambda row: [json.loads(row)])

        else:
            # Column is now array of strings (containing lists) -> convert to list of lists
            df_mean_auc_per_outer_fold[column] = df_mean_auc_per_outer_fold[column].map(lambda row: [json.loads(i) for i in row])

        # Flatten list of lists to 1 list with integers
        df_mean_auc_per_outer_fold[column] = df_mean_auc_per_outer_fold[column].map(lambda row: [item for sublist in row for item in sublist])

    # Corresponding to the highest mean auc with the mean probabilities predictions
    auc_mean_over_all_folds = df_mean_auc_per_outer_fold.loc[df_mean_auc_per_outer_fold.groupby(['outer_fold'])['mean_prob_auc'].idxmax()].copy().reset_index(drop=True)['mean_prob_auc'].mean()

    # Corresponding to the highest mean auc with the max probabilities predictions
    auc_max_over_all_folds = df_mean_auc_per_outer_fold.loc[df_mean_auc_per_outer_fold.groupby(['outer_fold'])['max_prob_auc'].idxmax()].copy().reset_index(drop=True)['max_prob_auc'].mean()

    # Here we check if either the max or mean probabilities predictions resulted in a higher AUC
    if auc_mean_over_all_folds > auc_max_over_all_folds:
        df_final = df_mean_auc_per_outer_fold.loc[df_mean_auc_per_outer_fold.groupby(['outer_fold'])['mean_prob_auc'].idxmax()].copy().reset_index(drop=True)
    else:
        df_final = df_mean_auc_per_outer_fold.loc[df_mean_auc_per_outer_fold.groupby(['outer_fold'])['max_prob_auc'].idxmax()].copy().reset_index(drop=True)

    # Here we select the best params corresponding to the highest AUC score for a specific fold.
    if not add_static_data:
        best_params = ast.literal_eval(df_final.loc[df_final['outer_fold'] == outer_fold_i, 'params'][outer_fold_i])
    if add_static_data:
        best_params = df_final.loc[df_final['outer_fold'] == outer_fold_i, 'params'][outer_fold_i]

    rec_ids_train_outer_fold = df_final.loc[df_final['outer_fold'] == outer_fold_i, 'rec_ids_train_outer'][outer_fold_i]
    rec_ids_test_outer_fold = df_final.loc[df_final['outer_fold'] == outer_fold_i, 'rec_ids_test_outer'][outer_fold_i]
    rec_ids_train_inner_fold = df_final.loc[df_final['outer_fold'] == outer_fold_i, 'rec_ids_train_inner'][outer_fold_i]
    rec_ids_test_inner_fold = df_final.loc[df_final['outer_fold'] == outer_fold_i, 'rec_ids_test_inner'][outer_fold_i]

    # Safety check that train and test set of outer fold are mutually exclusive
    assert not set(rec_ids_train_outer_fold) == set(rec_ids_test_outer_fold)

    # Safety check that train+test of inner and test set of outer fold are mutually exclusive
    assert not set(rec_ids_test_outer_fold) == set(rec_ids_train_inner_fold + rec_ids_test_inner_fold)

    return best_params, rec_ids_train_outer_fold, rec_ids_test_outer_fold


def main(model_name: str):
    cross_validation_final_train(model_name)


if __name__ == "__main__":
    out_path_model = os.path.abspath("trained_models")
    hyperopt_path = os.path.abspath("hyperparameters")

    # Command line arguments
    parser = argparse.ArgumentParser(description='Train a final model (on the train+validation dataset) using the '
                                                 'optimal hyperparameters obtained after running optimization.py. The '
                                                 'optimal hyperparameters have to be put in the best_params.json file '
                                                 'and the optional_model part has to be put in the optional_model_dict '
                                                 'and bidirectional_lstm_dict, which are placed at the top of this '
                                                 'final_train.py file. The final model will be saved in the '
                                                 'out_path_model folder, which you have to specify in this main file. '
                                                 'After running this file, you have to put the name of your final '
                                                 'models in the final_models.json file and then you can run '
                                                 'evaluation.py.')

    parser.add_argument('--model', type=str, required=True, help="Select what model to use: 'lstm' or 'tcn'",
                        choices=['tcn', 'lstm'])

    parser.add_argument('--feature_name', type=str, required=True,
                        help="Select what feature to use for data reduction: 'sample_entropy', 'peak_frequency' or "
                             "'median_frequency'",
                        choices=['sample_entropy', 'peak_frequency', 'median_frequency'])

    parser.add_argument('--hyperoptimization_file_name', type=str, required=True,
                        help="Name of the file with the results of the hyperoptimization run. The best params for each "
                             "model are already provided in the `best_params.json` file.")

    parser.add_argument('--n_folds', type=int, required=True, help="Number of outer folds used. Default is 5.")

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
                        help="Add static clinical data to the model. Use either the --add_static_data or the"
                             "--no_static_data flag")
    parser.add_argument('--no_static_data', dest='add_static_data', action='store_false',
                        required=('--model' in sys.argv and '--add_static_data' not in sys.argv),
                        help="Use only the EHG data for modeling. Use either the --add_static_data or the"
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

    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

    features_to_use = ['channel_1_filt_0.34_1_hz', 'channel_2_filt_0.34_1_hz', 'channel_3_filt_0.34_1_hz']
    fs = 20

    main(FLAGS.model)
