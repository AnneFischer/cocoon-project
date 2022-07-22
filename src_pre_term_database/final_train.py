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

settings_path = '/Users/AFischer/PycharmProjects/cocoon-project/references/settings'

file_paths = read_settings(settings_path, 'file_paths')
data_path = file_paths['data_path']


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

    if model_name == 'tcn_feature_sequence':
        final_train_tcn_feature_sequence(X_train, X_val, df_signals, df_clinical_information, df_static_information,
                                         best_params, features_to_use, num_static_features=num_static_features,
                                         add_static_data=add_static_data, copies=copies)

    elif model_name == 'lstm_feature_sequence':
        final_train_lstm_feature_sequence(X_train, X_val, df_signals, df_clinical_information, df_static_information,
                                          best_params, features_to_use, num_static_features=num_static_features,
                                          add_static_data=add_static_data)


if __name__ == "__main__":
    # Model name is either 'tcn_feature_sequence' or 'lstm_feature_sequence'
    model_name = 'tcn_feature_sequence'
    # Name of the file where the results of the hyperparameter search is saved
    optimal_params_file_name = 'tcn_data_trials_feature_median_frequency_combined_2022-05-08_14-19.csv'
    out_path_model = '/Users/AFischer/Documents/PhD_onderzoek/term_preterm_database/output/model'
    path_to_optimal_params = f'{out_path_model}/hyper_parameter_opt/{optimal_params_file_name}'

    hyper_opt_params = pd.read_csv(f'{path_to_optimal_params}')
    # eval does not work for the case of LSTM with static data, so you need to hard copy the best params for these
    # cases
    best_params = eval(hyper_opt_params.sort_values(by=['loss']).reset_index()['params'][0])

    features_to_use = ['channel_1_filt_0.34_1_hz', 'channel_2_filt_0.34_1_hz', 'channel_3_filt_0.34_1_hz']
    add_static_data = True
    copies = True
    main(model_name, features_to_use, best_params, add_static_data, copies)
