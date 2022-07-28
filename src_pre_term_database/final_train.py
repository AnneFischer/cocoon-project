import pandas as pd
from src_pre_term_database.optimization import OptimizationTCNFeatureSequence, \
    OptimizationStatefulFeatureSequenceLSTM, OptimizationCombinedLSTM, OptimizationTCNFeatureSequenceCombined, \
    OptimizationTCNFeatureSequenceCombinedCopies
from src_pre_term_database.modeling import TCN, LSTMStatefulClassificationFeatureSequence, \
    LSTMCombinedModel, TCNCombinedModel, TCNCombinedModelCopies
from src_pre_term_database.load_dataset import build_clinical_information_dataframe, build_demographics_dataframe
from src_pre_term_database.data_processing_and_feature_engineering import train_val_test_split, preprocess_static_data
from src_pre_term_database.utils import read_settings
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict
import constants as c
import argparse
import sys
import os

settings_path = os.path.abspath("references/settings")

file_paths = read_settings(settings_path, 'file_paths')
data_path = file_paths['data_path']

optional_model_dict = {
    "lstm_sample_entropy_with_static_data":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(36, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU(), nn.Linear(in_features=36, out_features=13, bias=True), nn.ReLU()),
         'bidirectional': True},
    "lstm_peak_frequency_with_static_data":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True), nn.ReLU(),
                                         nn.Linear(in_features=29, out_features=18, bias=True),
                                         nn.ReLU()),
         'bidirectional': True},
    "lstm_median_frequency_with_static_data":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(36, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU()),
         'bidirectional': True},
    "tcn_sample_entropy_with_static_data":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(21, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU())},
    "tcn_peak_frequency_with_static_data":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(28, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU())},
    "tcn_median_frequency_with_static_data":
        {'optional_model': nn.Sequential(nn.BatchNorm1d(31, eps=1e-05, momentum=0.1, affine=True,
                                                        track_running_stats=True),
                                         nn.ReLU())}
}

bidirectional_lstm_dict = {
    "lstm_sample_entropy": {'bidirectional': True},
    "lstm_peak_frequency": {'bidirectional': True},
    "lstm_median_frequency": {'bidirectional': True},
    "lstm_sample_entropy_with_static_data": {'bidirectional': True},
    "lstm_peak_frequency_with_static_data": {'bidirectional': True},
    "lstm_median_frequency_with_static_data": {'bidirectional': True}
}


def final_train_lstm_feature_sequence(X_train: pd.DataFrame, X_val: pd.DataFrame, df_signals: pd.DataFrame,
                                      df_clinical_information: pd.DataFrame, df_static_information: pd.DataFrame,
                                      best_params: Dict, features_to_use: List[str], num_static_features: int,
                                      add_static_data: bool = False):
    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)

    n_classes = 1
    input_channels = len(features_to_use)
    reduced_sequence_length = 50
    sub_sequence_length = 10
    # The num_sub_sequences variable is the number of sub_sequences necessary to complete an entire sequence
    num_sub_sequences = int(reduced_sequence_length / sub_sequence_length)
    device = 'cpu'

    if not add_static_data:
        model_lstm = LSTMStatefulClassificationFeatureSequence(input_size=input_channels,
                                                               hidden_size=best_params['hidden_dim'],
                                                               num_layers=best_params['layer_dim'],
                                                               dropout=best_params['drop_out_lstm'],
                                                               output_size=n_classes,
                                                               bidirectional=best_params['bidirectional'],
                                                               batch_size=best_params['batch_size'],
                                                               device=device, batch_first=True)

    elif add_static_data:
        model_lstm = LSTMCombinedModel(input_dim_seq=input_channels, hidden_dim_seq=best_params['hidden_dim_seq'],
                                       input_dim_static=num_static_features,
                                       hidden_dim_static=best_params['hidden_dim_static'],
                                       layer_dim=best_params['layer_dim'], bidirectional=best_params['bidirectional'],
                                       batch_size=best_params['batch_size'], output_dim=n_classes,
                                       model_optional=best_params['optional_model'],
                                       dropout_prob=best_params['drop_out_lstm'], device=device)

    model_lstm.to(device)

    # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
    # https://discuss.pytorch.org/t/bcewithlogitsloss-and-class-weights/88837
    pos_weight = torch.Tensor([((89 + 68) / 23)])

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model_lstm.parameters(), lr=best_params['learning_rate'],
                           weight_decay=best_params['weight_decay'])

    if not add_static_data:
        opt_model_lstm_stateful_feature = OptimizationStatefulFeatureSequenceLSTM(model=model_lstm,
                                                                                  loss_fn=loss_fn,
                                                                                  optimizer=optimizer,
                                                                                  num_sub_sequences=num_sub_sequences,
                                                                                  device=device)

        opt_model_lstm_stateful_feature.final_train(None, df_signals, df_clinical_information, df_static_information,
                                                    X_train_val, best_params, features_to_use,
                                                    feature_name=best_params['feature_name'],
                                                    add_static_data=add_static_data,
                                                    n_epochs=best_params['num_epochs'])

    if add_static_data:
        opt_lstm = OptimizationCombinedLSTM(model=model_lstm, loss_fn=loss_fn, optimizer=optimizer,
                                            num_sub_sequences=num_sub_sequences, device=device)

        opt_lstm.final_train(None, df_signals, df_clinical_information, df_static_information, X_train_val, best_params,
                             features_to_use, feature_name=best_params['feature_name'], add_static_data=add_static_data,
                             model_optional=best_params['optional_model'], n_epochs=best_params['num_epochs'])


def final_train_tcn_feature_sequence(X_train: pd.DataFrame, X_val: pd.DataFrame, df_signals: pd.DataFrame,
                                     df_clinical_information: pd.DataFrame, df_static_information: pd.DataFrame,
                                     best_params: Dict, features_to_use: List[str], num_static_features: int,
                                     add_static_data: bool = False, copies: bool = False):
    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)

    n_classes = 1
    reduced_sequence_length = 50
    sub_sequence_length = 10
    # The num_sub_sequences variable is the number of sub_sequences necessary to complete an entire sequence
    num_sub_sequences = int(reduced_sequence_length / sub_sequence_length)
    device = 'cpu'

    channel_sizes = [best_params['num_hidden_units_per_layer']] * best_params['num_levels']

    if add_static_data and not copies:
        input_channels = len(features_to_use)
        input_dim_static = num_static_features
        model_tcn = TCNCombinedModel(input_channels, n_classes, channel_sizes, stride=best_params['stride'],
                                     kernel_size=best_params['kernel_size'], dropout=best_params['drop_out'],
                                     input_dim_static=input_dim_static,
                                     hidden_dim_combined=best_params['num_hidden_units_per_layer'] + best_params['hidden_dim_static'],
                                     model_optional=best_params['optional_model'])

    elif add_static_data and copies:
        input_channels = len(features_to_use) + num_static_features
        model_tcn = TCNCombinedModelCopies(input_channels, n_classes, channel_sizes, stride=best_params['stride'],
                                           kernel_size=best_params['kernel_size'], dropout=best_params['drop_out'],
                                           hidden_dim_combined=best_params['num_hidden_units_per_layer'],
                                           model_optional=best_params['optional_model'])

    elif not add_static_data:
        input_channels = len(features_to_use)
        model_tcn = TCN(input_channels, n_classes, channel_sizes, stride=best_params['stride'],
                        kernel_size=best_params['kernel_size'], dropout=best_params['drop_out'])

    model_tcn.to(device)

    # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
    # https://discuss.pytorch.org/t/bcewithlogitsloss-and-class-weights/88837
    pos_weight = torch.Tensor([((89 + 68) / 23)])

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model_tcn.parameters(), lr=best_params['learning_rate'],
                           weight_decay=best_params['weight_decay'])

    if add_static_data and not copies:
        opt_tcn = OptimizationTCNFeatureSequenceCombined(model=model_tcn, loss_fn=loss_fn, optimizer=optimizer,
                                                         num_sub_sequences=num_sub_sequences, device=device)

    elif add_static_data and copies:
        opt_tcn = OptimizationTCNFeatureSequenceCombinedCopies(model=model_tcn, loss_fn=loss_fn, optimizer=optimizer,
                                                               num_sub_sequences=num_sub_sequences, device=device)

    elif not add_static_data:
        opt_tcn = OptimizationTCNFeatureSequence(model=model_tcn, loss_fn=loss_fn, optimizer=optimizer,
                                                 num_sub_sequences=num_sub_sequences, device=device)

    opt_tcn.final_train(None, df_signals, df_clinical_information, df_static_information, X_train_val, best_params,
                        features_to_use, feature_name=best_params['feature_name'], add_static_data=add_static_data,
                        n_epochs=best_params['num_epochs'])


def main(model_name: str, features_to_use: List[str], best_params: Dict, add_static_data: bool, copies: bool):
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

    # Count the number of static features
    if add_static_data:
        _, _, selected_columns_fit_static, _, _ = preprocess_static_data(df_static_information, X_train, X_train,
                                                                         threshold_correlation=0.85)

        num_static_features = len(selected_columns_fit_static)

    if not add_static_data:
        num_static_features = 0

    if model_name == 'tcn':
        final_train_tcn_feature_sequence(X_train, X_val, df_signals, df_clinical_information, df_static_information,
                                         best_params, features_to_use, num_static_features=num_static_features,
                                         add_static_data=add_static_data, copies=copies)

    elif model_name == 'lstm':
        final_train_lstm_feature_sequence(X_train, X_val, df_signals, df_clinical_information, df_static_information,
                                          best_params, features_to_use, num_static_features=num_static_features,
                                          add_static_data=add_static_data)


if __name__ == "__main__":
    out_path_model = os.path.abspath("trained_models")

    # Command line arguments
    parser = argparse.ArgumentParser(description='Train a final model (on the train+validation dataset) using the '
                                                 'optimal hyperparameters obtained afer running optimization.py. The '
                                                 'optimal hyperparameters have to be put in the best_params.json file '
                                                 'and the optional_model part has to be put in the optional_model_dict '
                                                 'and bidirectional_lstm_dict, which are placed at the top of this '
                                                 'final_train.py file. The final model will be saved in the '
                                                 'out_path_model folder, which you have to specify in this main file. '
                                                 'After running this file, you have to put the name of your final '
                                                 'model in the final_models.json file and then you can run '
                                                 'evaluation.py.')

    parser.add_argument('--model', type=str, required=True,
                        help="Select what model to use: 'lstm' or 'tcn'",
                        choices=['tcn', 'lstm'])

    parser.add_argument('--feature_name', type=str, required=True,
                        help="Select what feature to use for data reduction: 'sample_entropy', 'peak_frequency' or "
                             "'median_frequency'",
                        choices=['sample_entropy', 'peak_frequency', 'median_frequency'])

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

    if FLAGS.add_static_data:
        best_params_model_name = f'{FLAGS.model}_{FLAGS.feature_name}_with_static_data'

    elif not FLAGS.add_static_data:
        best_params_model_name = f'{FLAGS.model}_{FLAGS.feature_name}'

    best_params: dict = read_settings(settings_path, 'best_params')
    best_params = best_params[best_params_model_name]

    final_model: dict = read_settings(settings_path, 'final_models')
    final_model = final_model[best_params_model_name]

    # If static data is added to the model, add the optional model part to the best_params dict
    if FLAGS.add_static_data:
        best_params.update(optional_model_dict[best_params_model_name])

    # Add bidirectional as a boolean to the best_params dict for the LSTM models
    if FLAGS.model == 'lstm':
        best_params.update(bidirectional_lstm_dict[best_params_model_name])

    features_to_use = ['channel_1_filt_0.34_1_hz', 'channel_2_filt_0.34_1_hz', 'channel_3_filt_0.34_1_hz']

    main(FLAGS.model, features_to_use, best_params, FLAGS.add_static_data, FLAGS.use_copies_for_static_data)
