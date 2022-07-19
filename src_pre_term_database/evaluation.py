import pandas as pd
from src_pre_term_database.optimization import OptimizationTCNFeatureSequence, \
    OptimizationStatefulFeatureSequenceLSTM, OptimizationCombinedLSTM, OptimizationTCNFeatureSequenceCombined, \
    OptimizationTCNFeatureSequenceCombinedCopies
from src_pre_term_database.modeling import TCN, LSTMStatefulClassificationFeatureSequence, \
    LSTMCombinedModel, TCNCombinedModel, TCNCombinedModelCopies
from src_pre_term_database.load_dataset import build_signal_dataframe, build_clinical_information_dataframe, \
    build_demographics_dataframe
from src_pre_term_database.data_processing_and_feature_engineering import create_filtered_channels, \
    remove_first_n_samples_of_signals, remove_last_n_samples_of_signals, train_val_test_split, \
    generate_feature_data_loaders, preprocess_static_data
from src_pre_term_database.utils import read_settings
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Union
import constants as c
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import math

settings_path = '/Users/AFischer/PycharmProjects/cocoon-project/references/settings'

file_paths = read_settings(settings_path, 'file_paths')
data_path = file_paths['data_path']


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


def evaluate_tcn_feature_sequence(df_signals: pd.DataFrame, df_clinical_information: pd.DataFrame,
                                  df_static_information: pd.DataFrame, X_train_val: pd.DataFrame,
                                  X_test: pd.DataFrame, trained_model_file_name: str, features_to_use: List[str],
                                  best_params: Dict, add_static_data: bool, copies: bool, num_static_features: int):
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

    if not add_static_data:
        opt_tcn = OptimizationTCNFeatureSequence(model=model_tcn, loss_fn=loss_fn, optimizer=optimizer,
                                                 num_sub_sequences=num_sub_sequences, device=device)

    custom_test_loader_orig, rec_ids_test = generate_feature_data_loaders(None, df_signals, df_clinical_information,
                                                                          df_static_information,
                                                                          X_train_val, X_test, best_params,
                                                                          columns_to_use=features_to_use,
                                                                          feature_name=best_params['feature_name'],
                                                                          reduced_seq_length=reduced_sequence_length,
                                                                          sub_seq_length=sub_sequence_length,
                                                                          fs=20, shuffle=False,
                                                                          add_static_data=add_static_data,
                                                                          test_phase=True)

    out_path_model = '/Users/AFischer/Documents/PhD_onderzoek/term_preterm_database/output/model'
    ck_point = torch.load(f'{out_path_model}/{trained_model_file_name}', map_location=torch.device('cpu'))

    if not add_static_data:
        all_test_preds, all_test_probs, test_labels, results_dict = opt_tcn.evaluate(custom_test_loader_orig, ck_point)

    if add_static_data:
        all_test_preds, all_test_probs, test_labels, results_dict = opt_tcn.evaluate(custom_test_loader_orig,
                                                                                     features_to_use, ck_point)

    df_interval_tcn_preds = create_interval_matrix(all_test_preds, rec_ids_test, num_sub_sequences=num_sub_sequences)
    df_interval_tcn_probs = create_interval_matrix(all_test_probs, rec_ids_test, num_sub_sequences=num_sub_sequences)

    results_dict = calculate_auc_ap_second_largest_values(df_interval_tcn_preds, df_interval_tcn_probs,
                                                          test_labels, results_dict)

    return all_test_preds, all_test_probs, rec_ids_test, results_dict


def evaluate_lstm_feature_sequence(df_signals: pd.DataFrame, df_clinical_information: pd.DataFrame,
                                   df_static_information: pd.DataFrame, X_train_val: pd.DataFrame,
                                   X_test: pd.DataFrame, trained_model_file_name: str, features_to_use: List[str],
                                   best_params: Dict, add_static_data: bool, num_static_features: int):

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

    out_path_model = '/Users/AFischer/Documents/PhD_onderzoek/term_preterm_database/output/model'
    ck_point = torch.load(f'{out_path_model}/{trained_model_file_name}', map_location=torch.device('cpu'))

    custom_test_loader_orig, rec_ids_test = generate_feature_data_loaders(None, df_signals, df_clinical_information,
                                                                          df_static_information,
                                                                          X_train_val, X_test, best_params,
                                                                          columns_to_use=features_to_use,
                                                                          feature_name=best_params['feature_name'],
                                                                          reduced_seq_length=reduced_sequence_length,
                                                                          sub_seq_length=sub_sequence_length,
                                                                          fs=20, shuffle=False,
                                                                          add_static_data=add_static_data,
                                                                          test_phase=True)

    if not add_static_data:
        opt_model_lstm_stateful_feature = OptimizationStatefulFeatureSequenceLSTM(model=model_lstm,
                                                                                  loss_fn=loss_fn,
                                                                                  optimizer=optimizer,
                                                                                  num_sub_sequences=num_sub_sequences,
                                                                                  device=device)
        all_test_preds, all_test_probs, test_labels, results_dict = opt_model_lstm_stateful_feature.evaluate(
            custom_test_loader_orig, ck_point)

    if add_static_data:
        opt_lstm = OptimizationCombinedLSTM(model=model_lstm, loss_fn=loss_fn, optimizer=optimizer,
                                            num_sub_sequences=num_sub_sequences, device=device)

        all_test_preds, all_test_probs, test_labels, results_dict = opt_lstm.evaluate(custom_test_loader_orig,
                                                                                      best_params['optional_model'],
                                                                                      features_to_use, ck_point)

    df_interval_lstm_preds = create_interval_matrix(all_test_preds, rec_ids_test,
                                                    num_sub_sequences=num_sub_sequences)
    df_interval_lstm_probs = create_interval_matrix(all_test_probs, rec_ids_test,
                                                    num_sub_sequences=num_sub_sequences)

    results_dict = calculate_auc_ap_second_largest_values(df_interval_lstm_preds, df_interval_lstm_probs,
                                                          test_labels, results_dict)

    return all_test_preds, all_test_probs, rec_ids_test, results_dict


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


def main(trained_model_file_name: str, features_to_use: List[str], best_params: Dict, model_name: str,
         add_static_data: bool, copies: bool):
    # df_signals = build_signal_dataframe(data_path, settings_path)
    # df_signals_new = create_filtered_channels(df_signals, ['channel_1', 'channel_2', 'channel_3'],
    #                                           [[0.34, 1], [0.08, 4], [0.3, 3], [0.3, 4]], fs=20, order=4)
    # df_signals_new = remove_first_n_samples_of_signals(df_signals_new, n=3600)
    # df_signals_new = remove_last_n_samples_of_signals(df_signals_new, n=3600)
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
    if add_static_data:
        _, _, selected_columns_fit_static, _, _ = preprocess_static_data(df_static_information, X_train, X_train,
                                                                         threshold_correlation=0.85)

        num_static_features = len(selected_columns_fit_static)

    if not add_static_data:
        num_static_features = 0

    if model_name == 'tcn_feature_sequence':
        all_test_preds, all_test_probs, rec_ids_test, results_dict = evaluate_tcn_feature_sequence(df_signals,
                                                                                                   df_clinical_information,
                                                                                                   df_static_information,
                                                                                                   X_train_val, X_test,
                                                                                                   trained_model_file_name,
                                                                                                   features_to_use,
                                                                                                   best_params,
                                                                                                   add_static_data=add_static_data,
                                                                                                   copies=copies,
                                                                                                   num_static_features=num_static_features)

    elif model_name == 'lstm_feature_sequence':
        all_test_preds, all_test_probs, rec_ids_test, results_dict = evaluate_lstm_feature_sequence(df_signals,
                                                                                                    df_clinical_information,
                                                                                                    df_static_information,
                                                                                                    X_train_val, X_test,
                                                                                                    trained_model_file_name,
                                                                                                    features_to_use,
                                                                                                    best_params,
                                                                                                    add_static_data=add_static_data,
                                                                                                    num_static_features=num_static_features)

    return all_test_preds, all_test_probs, rec_ids_test, results_dict


if __name__ == "__main__":
    model_name = 'tcn_feature_sequence'
    out_path_model = '/Users/AFischer/Documents/PhD_onderzoek/term_preterm_database/output/model'
    trained_model_file_name = '2022-04-05_15-09_best_model_tcn_feature_seq_final_train.pth'
    hyper_opt_results_file_name = 'tcn_data_trials_feature_peak_frequency_2022-04-04_22-11.csv'
    path_to_optimal_params = f'{out_path_model}/hyper_parameter_opt/'

    hyper_opt_params = pd.read_csv(f'{path_to_optimal_params}')

    # eval does not work in the case of a LSTM model with added static data,
    # you have to hard copy the best_params in that case
    best_params = eval(hyper_opt_params.sort_values(by=['loss']).reset_index()['params'][0])

    add_static_data = False
    copies = True
    features_to_use = ['channel_1_filt_0.34_1_hz', 'channel_2_filt_0.34_1_hz', 'channel_3_filt_0.34_1_hz']

    all_test_preds, all_test_probs, rec_ids_test, results_dict = main(trained_model_file_name, features_to_use,
                                                                      best_params, model_name=model_name,
                                                                      add_static_data=add_static_data, copies=copies)

    results_dict.update({'model_file_name': trained_model_file_name,
                         'hyper_opt_file_name': hyper_opt_results_file_name})
    print(results_dict)
