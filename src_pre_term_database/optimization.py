import pandas as pd
import sys
from src_pre_term_database.modeling import TCN, get_num_levels_based_on_receptive_field, \
    LSTMStatefulClassificationFeatureSequence, LSTMCombinedModel, TCNCombinedModel, TCNCombinedModelCopies
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src_pre_term_database.data_processing_and_feature_engineering import preprocess_signal_data, generate_dataloader, \
    add_static_data_to_signal_data, basic_preprocessing_static_data, basic_preprocessing_signal_data
from src_pre_term_database.load_dataset import build_clinical_information_dataframe
import constants as c
import matplotlib.pyplot as plt
import datetime
import optuna
import joblib
from src_pre_term_database.utils import read_settings, convert_columns_to_numeric
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, \
    average_precision_score
import csv
from timeit import default_timer as timer
import math
from typing import List, Dict
import argparse
import os
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader


settings_path = os.path.abspath("references/settings")

file_paths = read_settings(settings_path, 'file_paths')
data_path = file_paths['data_path']


class VariablesChangeException(Exception):
  pass


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Taken from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py with a
    small adjustment (don't save model when validation loss decreases)
    """
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0


class OptimizationLSTM:
    """Train, optimize and evaluate a LSTM model. Batches are processed independent of each other, meaning that the
    hidden cells and states are resetted to zero after each batch has been processed.

    Parameters
    ----------
    model : nn.Module
        LSTM model with specified input_dim, hidden_dim, output_dim, batch_size, device, layer_dim, batch_first.
    loss_fn : torch.nn.modules.loss
        Loss function
    optimizer : torch.optim
        Optimizer function.
    device : str
        'cpu' or 'cuda'.
    """
    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_predictions = []
        self.val_predictions = []
        self.test_predictions = []
        self.train_probabilities = []
        self.val_probabilities = []
        self.test_probabilities = []
        self.test_labels = []

    def get_predictions_binary_model(self, y_pred, train, val, test):
        """Get the predictions and probabilities for the train/val/test batch"""
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, you are " \
                                                     f"now in {len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        # Calculate the probabilities from the raw logits
        output_prob = torch.sigmoid(y_pred)

        pred = output_prob > 0.5  # apply threshold to get class predictions

        # Data is first moved to cpu and then converted to numpy array
        if train:
            self.train_predictions.append(pred.cpu().data.numpy())
            self.train_probabilities.append(output_prob.cpu().data.numpy())
        elif val:
            self.val_predictions.append(pred.cpu().data.numpy())
            self.val_probabilities.append(output_prob.cpu().data.numpy())
        elif test:
            self.test_predictions.append(pred.cpu().data.numpy())
            self.test_probabilities.append(output_prob.cpu().data.numpy())

    def check_if_only_one_boolean(self, *args):
        """Method to check if only one variable in a list of variables has a True value.
        Since Booleans are a subtype of plain integers, you can sum the list of integers
        quite easily and you can also pass true booleans into this function as well.
        """
        return sum(args) == 1

    def train_step(self, x, y):
        """Returns loss and prediction."""
        # Sets model to train mode
        self.model.train()

        # Zeroes gradients
        self.optimizer.zero_grad()

        # Forward pass to get output/logits. For BCEwithlogits loss, y_pred will be logits
        y_pred = self.model(x)

        # Computes loss (input for BCEwithlogits loss should be (logits, true label))
        # The BCEwithlogits loss combines a Sigmoid layer and the BCELoss in one single class.
        # Therefore, you have to add logits as input (and not put the input through a sigmoid first)
        loss = self.loss_fn(y_pred, y)

        a = list(self.model.parameters())[0].clone()

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updates parameters
        self.optimizer.step()

        b = list(self.model.parameters())[0].clone()

        assert not torch.equal(a.data, b.data)

        print(list(self.model.parameters())[0].grad)

        return loss, y_pred

    def train(self, train_loader, val_loader, trial, n_epochs=50):
        output_path = file_paths['output_path'] + "/" + "model"
        current_date_and_time = "{:%Y-%m-%d_%H-%M}".format(datetime.datetime.now())
        file_name_output = f'{current_date_and_time}_best_model_stateless'

        for epoch in range(1, n_epochs + 1):
            total_train_loss_epoch = 0.0
            for t, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # avg_train_batch_loss contains the value of the average cost of all the training
                # examples of the current batch
                avg_train_batch_loss, y_pred = self.train_step(x_batch, y_batch)

                self.get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                total_train_loss_epoch += (avg_train_batch_loss.item() * x_batch.size(0))

            # Calculate average sample train loss for this particular epoch.
            avg_sample_train_loss_epoch = total_train_loss_epoch / len(train_loader.sampler)
            self.train_losses.append(avg_sample_train_loss_epoch)

            self.train_probabilities.clear()
            self.train_predictions.clear()

            self.model.eval()
            with torch.no_grad():
                total_val_loss_epoch = 0.0
                correct_val, total_samples_val = 0, 0
                for t, (x_val, y_val) in enumerate(val_loader):
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    # forward pass: compute predicted outputs by passing inputs to the model
                    y_pred = self.model(x_val)  # output shape is [batch_size, 1] and contains logits

                    self.get_predictions_binary_model(y_pred, train=False, val=True, test=False)

                    total_samples_val += y_val.size(0)

                    avg_val_batch_loss = self.loss_fn(y_pred, y_val)
                    # total_val_loss_epoch accumulates the total loss per validation batch for the entire epoch
                    # update running training loss
                    total_val_loss_epoch += (avg_val_batch_loss.item() * x_val.size(0))

                # Calculate average sample validation loss for this particular epoch.
                avg_sample_val_loss_epoch = total_val_loss_epoch / len(val_loader.sampler)
                self.val_losses.append(avg_sample_val_loss_epoch)

                self.val_probabilities.clear()
                self.val_predictions.clear()

                if not trial:
                    self.save_best_model(avg_sample_val_loss_epoch, epoch, self.model, self.optimizer,
                                         self.loss_fn, output_path, file_name_output)
                if trial:
                    # Add prune mechanism
                    trial.report(avg_sample_val_loss_epoch, epoch)

            print(
                f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.10f}\t "
                f"Validation loss: {avg_sample_val_loss_epoch:.10f}"
            )
            print('-' * 50)

        self.save_loss_plot(current_date_and_time)
        print('TRAINING COMPLETE')

        return avg_sample_val_loss_epoch

    def final_train(self, train_val_loader, n_epochs=50):
        output_path = file_paths['output_path'] + "/" + "model"

        current_date_and_time = "{:%Y-%m-%d_%H-%M}".format(datetime.datetime.now())
        file_name_output = f'{current_date_and_time}_model_sample_entropy_final_train'

        for epoch in range(1, n_epochs + 1):
            total_train_loss_epoch = 0.0
            for t, (x_batch, y_batch) in enumerate(train_val_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # avg_train_batch_loss contains the value of the average cost of all the training
                # examples of the current batch
                avg_train_batch_loss, y_pred = self.train_step(x_batch, y_batch)

                self.get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                total_train_loss_epoch += (avg_train_batch_loss.item() * x_batch.size(0))

            # Calculate average sample train loss for this particular epoch.
            avg_sample_train_loss_epoch = total_train_loss_epoch / len(train_val_loader.sampler)
            self.train_losses.append(avg_sample_train_loss_epoch)

            self.train_probabilities.clear()
            self.train_predictions.clear()

            print(f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.10f}")
            print('-' * 50)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn,
        }, f'{output_path}/{file_name_output}.pth')
        print('TRAINING COMPLETE')

    def evaluate(self, test_loader, checkpoint):
        model_epoch = checkpoint['epoch']
        print(f"Model was saved at {model_epoch} epochs\n")

        print(f'Loading at epoch {model_epoch} saved model weights...')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        with torch.no_grad():
            correct_test = 0
            for x_test, y_test in test_loader:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                y_pred = self.model(x_test)  # output shape is [batch_size, 1] and contains logits

                self.get_predictions_binary_model(y_pred, train=False, val=False, test=True)
                self.test_labels.append(y_test.cpu().data.numpy())
                self.test_labels = list(np.array(self.test_labels, dtype=np.int64).flat)
                self.test_predictions = list(np.array(self.test_predictions, dtype=np.int64).flat)

                print(f'Test labels: {self.test_labels}')
                print(f'Test predictions: {self.test_predictions}')

                num_correct_batch = len([i for i, j in zip(self.test_predictions, self.test_labels) if i == j])
                print(f'The number of correct predicted samples: {num_correct_batch}/{len(y_test)}')

                correct_test += num_correct_batch
                precision_test, recall_test, _ = precision_recall_curve(list(np.array(self.test_labels).flat),
                                                                        list(np.array(self.test_probabilities).flat))
                self.plot_prec_recall_curve(recall_test, precision_test)

                print(f'Precision score: {precision_score(self.test_labels, self.test_predictions)}')
                print(f'Recall score: {recall_score(self.test_labels, self.test_predictions)}')
                print(f'F1 score: {f1_score(self.test_labels, self.test_predictions)}')
                print(f'Average precision score: '
                      f'{average_precision_score(self.test_labels, list(np.array(self.test_probabilities).flat))}')
                print(f'AUC score: {roc_auc_score(self.test_labels, list(np.array(self.test_probabilities).flat))}')
                # We clear the list with the predictions for the next batch which contains the sequences
                # of new rec ids
                self.test_labels.clear()
                self.test_probabilities.clear()
                self.test_predictions.clear()

    def save_loss_plot(self, current_date_and_time):
        """Function to save the loss plot to disk."""
        output_path = file_paths['output_path'] + "/" + "model"
        file_name_output = f'{current_date_and_time}_loss.png'

        # loss plot
        plt.figure(figsize=(10, 7))
        plt.plot(
            self.train_losses, color='orange', linestyle='-',
            label='Train loss'
        )
        plt.plot(
            self.val_losses, color='red', linestyle='-',
            label='Validation loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # plt.savefig(f'{output_path}/{file_name_output}')
        plt.show()
        plt.close()

    def plot_prec_recall_curve(self, recall, precision):
        """Function to plot the precision recall curve."""
        plt.plot(recall, precision, marker='.', label='Test set')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()
        plt.close()


class OptimizationCombinedLSTM(OptimizationLSTM):
    """Train, optimize and evaluate a LSTM model that uses both static and sequential data. Batches
    are processed independent of each other, meaning that the hidden cells and states are resetted to
    zero after each batch has been processed.

    Parameters
    ----------
    model : nn.Module
        LSTM model with specified input_dim, hidden_dim, output_dim, batch_size, device, layer_dim, batch_first.
    loss_fn : torch.nn.modules.loss
        Loss function.
    optimizer : torch.optim
        Optimizer function.
    device : str
        'cpu' or 'cuda'.
    """
    def __init__(self, model, loss_fn, optimizer, num_sub_sequences, device):
        super().__init__(model, loss_fn, optimizer, device)
        self.num_sub_sequences = num_sub_sequences
        self.train_predictions = []
        self.val_predictions = []
        self.test_predictions = []
        self.train_probabilities = []
        self.val_probabilities = []
        self.test_probabilities = []
        self.final_train_predictions = []
        self.final_val_predictions = []
        self.final_test_predictions = []
        self.final_train_prob = []
        self.final_val_prob = []
        self.final_test_prob = []

        self.final_max_val_probabilities = []
        self.final_max_test_predictions = []
        self.final_max_test_probabilities = []
        self.train_losses = []
        self.val_losses = []
        self.val_labels = []
        self.test_labels = []

    def reset_states(self, batch_size):
        """Reset the hidden states and cells to zero."""
        for layer in self.model.modules():
            if hasattr(layer, 'reset_hidden_cell'):
                layer.reset_hidden_cell(batch_size)
            if hasattr(layer, 'reset_hidden_state'):
                layer.reset_hidden_state(batch_size)

    def get_mean_prediction_and_probability(self, train, val, test):
        """Get the mean pred and prob over all sub-sequences.

        If a total sequence is 50 time steps and the sub_sequence length is 10 time steps,
        then there are 50 / 10 = 5 sub_sequences. Meaning, that we'll take the mean over the
        5 sub-sequences to obtain the mean pred and prob.

        Returns
        -------
        mean_pred : List[int]
            List of size [batch_size]. Containing the mean prediction for all batch_size rec ids.
        mean_prob : List[float]
            List of size [batch_size]. Containing the mean probability for all batch_size rec ids.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, you " \
                                                     f"are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        if train:
            predictions = self.train_predictions
            probabilities = self.train_probabilities
        elif val:
            predictions = self.val_predictions
            probabilities = self.val_probabilities
        elif test:
            predictions = self.test_predictions
            probabilities = self.test_probabilities

        # Calculate the mean prediction by taking the average over all sub_sequences that make up
        # one total sequence. The variable predictions contains all the sub_sequence predictions
        # belonging to batch_size rec ids.

        mean_pred = list(np.array(np.mean(np.stack(predictions, axis=1), axis=1), dtype=np.int64).flat)
        mean_prob = list(np.array(np.mean(np.stack(probabilities, axis=1), axis=1)).flat)

        max_pred = list(np.array(np.max(np.stack(predictions, axis=1), axis=1)).flat)
        max_prob = list(np.array(np.max(np.stack(probabilities, axis=1), axis=1)).flat)

        return mean_pred, mean_prob, max_pred, max_prob

    def obtain_correct_classified_instances(self, t, train, val, test):
        """Obtain the number of correctly classified instances by taking the mean prediction over all
        sub sequences and compare it against the true label.

        Parameters
        ----------
        y : torch.Tensor
            Tensor containing the true labels. Shape [batch_size, output_size].
        t : int
            The time step in the data loader.
        train : Boolean
            If in train mode.
        val : Boolean
            If in validation mode.
        test : Boolean
            If in test mode.
        Returns
        -------
        num_correct : int
            Percentage of change in prediction within the total sequences.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, " \
                                                     f"you are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        if train:
            predictions = self.train_predictions
        elif val:
            predictions = self.val_predictions
        elif test:
            predictions = self.test_predictions

        assert len(predictions) == self.num_sub_sequences, f'The number of predictions on time step {t} ' \
                                                           f'should be {self.num_sub_sequences}, but ' \
                                                           f'is {len(predictions)}'

        mean_pred, mean_prob, max_pred, max_prob = self.get_mean_prediction_and_probability(train, val, test)

        return mean_pred, mean_prob, max_pred, max_prob

    def evaluate_after_entire_sequence(self, y, t, train, val, test):
        """
        After an entire sequence is processed, we want to know the average prediction over the entire
        sequence and use that as a final prediction for the entire sequence.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, " \
                                                     f"you are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"

        # Data is first moved to cpu and then converted to numpy array
        true_label = y.cpu().data.numpy()
        mean_pred, mean_prob, max_pred, max_prob = self.obtain_correct_classified_instances(t,
                                                                                            train=train,
                                                                                            val=val,
                                                                                            test=test)

        # We clear the list with the predictions for the next batch which contains the sequences of new
        # rec ids
        if train:
            self.final_train_predictions.append(mean_pred)
            self.train_predictions.clear()
            self.final_train_prob.append(mean_prob)
            self.train_probabilities.clear()

        elif val:
            self.val_labels.append(true_label)
            self.final_val_predictions.append(mean_pred)
            self.val_predictions.clear()
            self.final_val_prob.append(mean_prob)
            self.val_probabilities.clear()

            self.final_max_val_probabilities.append(max_prob)

        elif test:
            self.test_labels.append(true_label)
            self.final_test_predictions.append(mean_pred)
            self.test_predictions.clear()
            self.final_test_prob.append(mean_prob)
            self.test_probabilities.clear()
            self.final_max_test_predictions.append(max_pred)
            self.final_max_test_probabilities.append(max_prob)

        return mean_pred, mean_prob, max_pred, max_prob

    def train_step_combined_model(self, x_seq, x_static, y):
        """Returns loss and prediction."""
        # Sets model to train mode
        self.model.train()

        # Zeroes gradients
        self.optimizer.zero_grad()

        # Forward pass to get output/logits. For BCEwithlogits loss, y_pred will be logits
        y_pred = self.model(x_seq, x_static)

        # Computes loss (input for BCEwithlogits loss should be (logits, true label))
        loss = self.loss_fn(y_pred, y)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updates parameters
        self.optimizer.step()

        return loss, y_pred

    def train_combined_model(self, trial, train_loader_list: List[DataLoader], test_loader_list: List[DataLoader],
                             features_to_use, params, n_epochs=50):
        """The model is trained/validated/tested on each sub-sequence up until each entire
        sequence has been processed. The entire sequence length can be 50 time steps and the
        sub_sequence length can be 10 time steps for instance. We will make a prediction after each sub-sequence
        has been processed.

        This means that 1 batch contains 10 time steps of batch_size rec ids. As a result, we will
        reset all predictions after 5 batches have been processed. Then we will start processing
        the entire sequences of the next batch_size rec ids and again reset after 5 batches have
        been processed. This process is repeated up until the entire sequence of each rec id has
        been processed.

        Parameters
        ----------
        trial :
        train_loader_list : List[DataLoader]
        test_loader_list : List[DataLoader]
        params : dict
            Dictionary with the hyperparameters and its values.
        n_epochs : int
            Number of epochs for which you want to train your model.
        """
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(1, n_epochs + 1):
            total_train_loss_epoch = 0.0
            correct_train, total_samples_train = 0, 0
            # Clear at the beginning of every epoch
            self.final_train_prob.clear()
            self.final_val_prob.clear()
            self.val_labels.clear()

            self.train_losses.clear()
            self.val_losses.clear()

            self.final_val_predictions.clear()
            self.final_max_val_probabilities.clear()

            for train_loader in train_loader_list:
                for t, (x_batch, y_batch) in enumerate(train_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size_train = x_batch.shape[0]

                    total_samples_train += batch_size_train

                    # Split the sequential and static data.
                    # The seq data are the first len(features_to_use) columns and the static
                    # data is the other part
                    x_batch_seq = x_batch[:, :, 0:len(features_to_use)].to(self.device)

                    x_batch_static = x_batch[:, :, len(features_to_use):]
                    # The static data is now the same length as the number of time steps
                    # in the seq data. Meaning, if there are 10 time steps in the seq data,
                    # then the static data is also copied for 10 time steps.
                    # As we don't want copies of the static data along all time steps,
                    # but just 1 time step we keep the first value of the static data
                    # for each sample in the batch.
                    # This is a bit of a quick fix for now, it would be neater to not
                    # have the copies in the first place
                    x_batch_static = x_batch_static[:, 0, :].to(self.device)
                    y_batch = y_batch.to(self.device)

                    # We manually reset the hidden states and cells after an entire sequence is processed
                    if t % self.num_sub_sequences == 0:
                        print(f'The hidden states and cells are resetted at the beginning of epoch {epoch}, '
                              f'batch {t} of train_loader')
                        self.reset_states(batch_size=batch_size_train)

                    # avg_train_batch_loss contains the value of the average cost of all the training
                    # examples of the current batch
                    avg_train_batch_loss, y_pred = self.train_step_combined_model(x_batch_seq, x_batch_static,
                                                                                  y_batch)

                    super().get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                    # After an entire sequence is processed, we clear the predictions and probs
                    if len(self.train_predictions) == self.num_sub_sequences:
                        self.train_predictions.clear()
                        self.train_probabilities.clear()

                    # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                    total_train_loss_epoch += (np.nan_to_num(avg_train_batch_loss.item()) * batch_size_train)

                    # We do truncated BPTT as the model otherwise would have to backpropagate through the entire
                    # sequence, which will lead to very long training times + vanishing/exploding gradients
                    self.model.lstm.hidden_state = self.model.lstm.hidden_state.detach()
                    self.model.lstm.hidden_cell = self.model.lstm.hidden_cell.detach()

                # Calculate average sample train loss for this particular epoch.
                avg_sample_train_loss_epoch = total_train_loss_epoch / total_samples_train
                self.train_losses.append(avg_sample_train_loss_epoch)

            self.model.eval()
            torch.autograd.set_detect_anomaly(True)
            with torch.no_grad():
                total_val_loss_epoch = 0.0
                correct_val, total_samples_val = 0, 0
                all_val_probabilities = []
                for val_loader in test_loader_list:
                    for t, (x_val, y_val) in enumerate(val_loader):

                        # The order in the batch MUST be [batch_size, sequence length, num_features]
                        batch_size_val = x_val.shape[0]

                        x_val_seq = x_val[:, :, 0:len(features_to_use)].to(self.device)
                        x_val_static = x_val[:, :, len(features_to_use):]
                        x_val_static = x_val_static[:, 0, :].to(self.device)
                        y_val = y_val.to(self.device)

                        # We manually reset the hidden states and cells after an entire sequence is processed
                        if t % self.num_sub_sequences == 0:
                            print(f'The hidden states and cells are resetted at the beginning of epoch {epoch}, '
                                  f'batch {t} of val_loader')
                            self.reset_states(batch_size=batch_size_val)

                        total_samples_val += batch_size_val

                        # forward pass: compute predicted outputs by passing inputs to the model
                        # output shape is [batch_size, output_dim] and contains logits
                        y_pred = self.model(x_val_seq, x_val_static)

                        avg_val_batch_loss = self.loss_fn(y_pred, y_val)

                        self.get_predictions_binary_model(y_pred, train=False, val=True, test=False)

                        if len(self.val_predictions) == self.num_sub_sequences:
                            all_val_probabilities.append(self.val_probabilities.copy())
                            mean_pred, mean_prob, max_pred, max_prob = self.evaluate_after_entire_sequence(y_val, t,
                                                                                                           train=False,
                                                                                                           val=True,
                                                                                                           test=False)
                            # self.val_predictions.clear()
                            # self.val_probabilities.clear()

                        # total_val_loss_epoch accumulates the total loss per validation batch for the entire epoch
                        total_val_loss_epoch += (avg_val_batch_loss.item() * batch_size_val)

                # Calculate average sample validation loss for this particular epoch
                avg_sample_val_loss_epoch = total_val_loss_epoch / total_samples_val
                self.val_losses.append(avg_sample_val_loss_epoch)

                val_labels_final = list(np.array(np.concatenate([array for array in self.val_labels], axis=0),
                                                dtype=np.int64).flat)
                final_val_prob_final = list(np.array(np.concatenate([array for array in self.final_val_prob],
                                                                   axis=0)).flat)

                final_max_val_probabilities_final = list(
                    np.array(np.concatenate([array for array in self.final_max_val_probabilities], axis=0)).flat)

                auc_mean_val = roc_auc_score(val_labels_final, final_val_prob_final)
                auc_max_val = roc_auc_score(val_labels_final, final_max_val_probabilities_final)

            print(f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.4f}\t "
                  f"Validation loss: {avg_sample_val_loss_epoch:.4f}\t AUC mean val: {auc_mean_val}\t AUC max val: {auc_max_val}")
            print('-' * 50)

        print('TRAINING COMPLETE')

        return auc_mean_val, auc_max_val, params

    def final_train(self, train_loader_list, feature_name, add_static_data, fold_i, features_to_use,
                    model_optional, n_epochs=50):
        """The final model will be trained with the optimal hyperparameters on the train+val datset."""
        output_path = file_paths['output_path'] + "/" + "model"

        current_date_and_time = "{:%Y-%m-%d_%H-%M}".format(datetime.datetime.now())

        if add_static_data:
            file_name_output = f'{current_date_and_time}_fold_{fold_i}_best_model_lstm_feature_{feature_name}_combined_seq_final_train'

        if not add_static_data:
            file_name_output = f'{current_date_and_time}_fold_{fold_i}_best_model_lstm_feature_{feature_name}_seq_final_train'

        for epoch in range(1, n_epochs + 1):
            self.final_train_prob.clear()
            total_train_loss_epoch = 0.0
            correct_train, total_samples_train = 0, 0

            for train_loader in train_loader_list:
                for t, (x_batch, y_batch) in enumerate(train_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size = x_batch.shape[0]
                    total_samples_train += batch_size

                    # We manually reset the hidden states and cells after an entire sequence is processed
                    if t % self.num_sub_sequences == 0:
                        print(
                            f'The hidden states and cells are resetted at the beginning of epoch {epoch}, '
                            f'batch {t} of train_loader')
                        self.reset_states(batch_size=batch_size)

                    x_batch_seq = x_batch[:, :, 0:len(features_to_use)].to(self.device)
                    x_batch_static = x_batch[:, :, len(features_to_use):]
                    x_batch_static = x_batch_static[:, 0, :].to(self.device)
                    y_batch = y_batch.to(self.device)

                    # avg_train_batch_loss contains the value of the average cost of all the training
                    # examples of the current batch
                    avg_train_batch_loss, y_pred = self.train_step_combined_model(x_batch_seq, x_batch_static,
                                                                                  y_batch)

                    self.get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                    # After an entire sequence is processed, we want to know the average prediction over the entire
                    # sequence and use that as a final prediction for the entire sequence
                    if len(self.train_predictions) == self.num_sub_sequences:
                        self.train_predictions.clear()
                        self.train_probabilities.clear()

                    # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                    total_train_loss_epoch += (avg_train_batch_loss.item() * batch_size)

                    # We do truncated BPTT as the model otherwise would have to backpropagate through the entire
                    # sequence, which will lead to very long training times + vanishing/exploding gradients
                    self.model.lstm.hidden_state = self.model.lstm.hidden_state.detach()
                    self.model.lstm.hidden_cell = self.model.lstm.hidden_cell.detach()

            # Calculate average sample train loss for this particular epoch.
            avg_sample_train_loss_epoch = total_train_loss_epoch / total_samples_train
            self.train_losses.append(avg_sample_train_loss_epoch)

            print(f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.10f}")
            print('-' * 50)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn}, f'{output_path}/{file_name_output}.pth')
        print('TRAINING COMPLETE')
        print(f'Model is saved at: {output_path}/{file_name_output}.pth')

    def evaluate(self, test_loader, model_optional, features_to_use, checkpoint):
        """Evaluate the model by making predictions over each subsequence and after all
        subsequences for each rec id have been processed take the mean over all subsequence predictions."""

        all_test_predictions = []
        all_test_probabilities = []

        model_epoch = checkpoint['epoch']
        print(f"Model was saved at {model_epoch} epochs\n")
        print(f'Loading at epoch {model_epoch} saved model weights...')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        with torch.no_grad():
            correct_test, total_samples_test = 0, 0
            for data_loader in test_loader:
                for t, (x_test, y_test) in enumerate(data_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size_test = x_test.shape[0]

                    x_test_seq = x_test[:, :, 0:len(features_to_use)].to(self.device)
                    x_test_static = x_test[:, :, len(features_to_use):]
                    x_test_static = x_test_static[:, 0, :].to(self.device)
                    y_test = y_test.to(self.device)

                    total_samples_test += batch_size_test

                    # We manually reset the hidden states and cells after an entire sequence is processed
                    if t % self.num_sub_sequences == 0:
                        print(f'The hidden states and cells are resetted at the beginning batch {t} of test_loader')
                        self.reset_states(batch_size=batch_size_test)

                    # forward pass: compute predicted outputs by passing inputs to the model
                    # output shape is [batch_size, 1] and contains logits
                    y_pred = self.model(x_test_seq, x_test_static)

                    self.get_predictions_binary_model(y_pred, train=False, val=False, test=True)

                    # After the entire sequence has been processed we take the mean prediction of all sub-sequence
                    # predictions
                    if len(self.test_predictions) == self.num_sub_sequences:
                        all_test_predictions.append(self.test_predictions.copy())
                        all_test_probabilities.append(self.test_probabilities.copy())
                        mean_pred, mean_prob, max_pred, max_prob = self.evaluate_after_entire_sequence(y_test, t,
                                                                                                       train=False,
                                                                                                       val=False,
                                                                                                       test=True)

        self.test_labels = list(np.array(np.concatenate([array for array in self.test_labels], axis=0),
                                         dtype=np.int64).flat)
        self.final_test_prob = list(np.array(np.concatenate([array for array in self.final_test_prob], axis=0)).flat)
        self.final_test_predictions = list(np.array(np.concatenate([array for array in self.final_test_predictions],
                                                                   axis=0)).flat)

        self.final_max_test_predictions = list(
            np.array(np.concatenate([array for array in self.final_max_test_predictions], axis=0)).flat)
        self.final_max_test_probabilities = list(
            np.array(np.concatenate([array for array in self.final_max_test_probabilities], axis=0)).flat)

        print(f'Precision score with mean pred: {precision_score(self.test_labels, self.final_test_predictions)}')
        print(f'Recall score with mean pred: {recall_score(self.test_labels, self.final_test_predictions)}')
        print(f'F1 score with mean pred: {f1_score(self.test_labels, self.final_test_predictions)}')
        print(f'AUC score with mean prob: {roc_auc_score(self.test_labels, self.final_test_prob)}')
        print(f'AP score with mean prob: {average_precision_score(self.test_labels, self.final_test_prob)}')

        print(f'Precision score with max pred: {precision_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'Recall score with max pred: {recall_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'F1 score with max pred: {f1_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'AUC score with max prob: {roc_auc_score(self.test_labels, self.final_max_test_probabilities)}')
        print(f'AP score with max prob: {average_precision_score(self.test_labels, self.final_max_test_probabilities)}')

        results_dict = {f'precision_mean_pred': precision_score(self.test_labels, self.final_test_predictions),
                        f'recall_mean_pred': recall_score(self.test_labels, self.final_test_predictions),
                        f'f1_mean_pred': f1_score(self.test_labels, self.final_test_predictions),
                        f'auc_mean_prob': roc_auc_score(self.test_labels, self.final_test_prob),
                        f'ap_mean_prob': average_precision_score(self.test_labels, self.final_test_prob),
                        f'precision_max_pred': precision_score(self.test_labels, self.final_max_test_predictions),
                        f'recall_max_pred': recall_score(self.test_labels, self.final_max_test_predictions),
                        f'f1_max_pred': f1_score(self.test_labels, self.final_max_test_predictions),
                        f'auc_max_prob': roc_auc_score(self.test_labels, self.final_max_test_probabilities),
                        f'ap_max_prob': average_precision_score(self.test_labels, self.final_max_test_probabilities)}

        return all_test_predictions, all_test_probabilities, self.test_labels, results_dict


class OptimizationStatefulLSTM(OptimizationLSTM):
    """Train, optimize and evaluate a stateful LSTM model. Stateful means that the hidden
    states and cells are passed on to the next batch instead of being resetted to zero at
    the beginning of each batch. This is useful when the sequence is too long to be processed
    by the LSTM model all at once. We therefore split up the sequence in multiple sub-sequences.
    The number of sub-sequences needed to process an entire sequence is denoted by the parameter
    num_sub_sequences. The hidden cells and states are manually resetted to zero after num_sub_sequences
    batches have been processed.

    Each sequence has been padded or truncated to the same length. We will mask the padded data points
    such that it does not contribute to the loss function.

    The final prediction will be made by making a prediction for each sub-sequence and at the end take
    the mean over all sub-sequence predictions.

    Parameters
    ----------
    model : nn.Module
        LSTM model with specified input_dim, hidden_dim, output_dim, batch_size, device,
        layer_dim, batch_first.
    loss_fn : torch.nn.modules.loss
        Loss function
    optimizer : torch.optim
        Optimizer function.
    num_sub_sequences : int
        Number of sub-sequences that make up an entire sequence.
    """

    def __init__(self, model, loss_fn, optimizer, device, num_sub_sequences):
        super().__init__(model, loss_fn, optimizer, device)
        self.num_sub_sequences = num_sub_sequences
        self.final_train_predictions = []
        self.final_val_predictions = []
        self.final_test_predictions = []
        self.final_train_prob = []
        self.final_val_prob = []
        self.final_test_prob = []

    def reset_states(self):
        """Reset the hidden states and cells to zero."""
        for layer in self.model.modules():
            if hasattr(layer, 'reset_hidden_cell'):
                layer.reset_hidden_cell()
            if hasattr(layer, 'reset_hidden_state'):
                layer.reset_hidden_state()

    def add_nans_for_padded_sequences(self, list_with_predictions, batch_size):
        """Add NaN values for the predictions over the padded parts of the sequence. A padded part
        (with only values of zero) will have no 'real' prediction as these padded values are meaningless.
        However, we do need to fill up these 'fake' predictions with NaN values.
        These NaN values will be disregarded when taking the mean prediction over all sub-sequences.

        It is necessary to run this step, because otherwise we run into errors because for
        each rec id we need to have the same number of sub-predictions.
        """
        for i, pred_list in enumerate(list_with_predictions):
            if len(pred_list) == batch_size:
                continue
            else:
                remainder = batch_size - len(pred_list)
                remainder_list = [np.nan] * remainder
                pred_list = np.concatenate((pred_list, remainder_list), axis=0)
                list_with_predictions[i] = pred_list

        return list_with_predictions

    def analyze_predictions(self, train, val, test, batch_size):
        """Analyze the predictions by checking how often there is a change in prediction
        within the entire sequence.

        If a total sequence is 28800 time steps and the sub_sequence length is 200 time steps,
        then there are 28800 / 200 = 144 sub_sequences. Meaning, that within an entire sequence,
        there can be a maximum of 143 changes in prediction.

        Returns
        -------
        perc_changes_in_pred : float
            Percentage of change in prediction within the total sequences.
        mean_pred : torch.Tensor
            Tensor of size [batch_size]. Containing the mean prediction for all batch_size rec ids.
        """
        assert super().check_if_only_one_boolean(train, val,
                                                 test), f"You can only be in either one of train/val/test mode, you " \
                                                        f"are now in " \
                                                        f"{len([var for var in [train, val, test] if var == True])} " \
                                                        f"modes"
        if train:
            predictions = self.train_predictions
            predictions = self.add_nans_for_padded_sequences(predictions, batch_size)
        elif val:
            predictions = self.val_predictions
            predictions = self.add_nans_for_padded_sequences(predictions, batch_size)
        elif test:
            predictions = self.test_predictions
            predictions = self.add_nans_for_padded_sequences(predictions, batch_size)

        # Calculate the mean prediction by taking the average over all sub_sequences that make up
        # one total sequence. The variable predictions contains all the sub_sequence predictions
        # belonging to batch_size rec ids.
        mean_pred = np.nanmean(np.stack(predictions, axis=0), axis=0)

        # For each sub_sequence prediction we check whether the subsequent sub_sequence prediction is different
        # If a subsequent sub_sequence prediction is different, the difference will be non-zero
        changes_in_pred = np.diff(predictions, axis=0)

        # Concatenate all values into one list
        changes_in_pred = changes_in_pred.flatten()

        # Here we sum all the cases where there has been a change in prediction going from one sub_sequence
        # to the next one
        num_changes_in_pred = np.sum(changes_in_pred != 0)

        perc_changes_in_pred = (num_changes_in_pred / len(changes_in_pred)) * 100

        return perc_changes_in_pred, mean_pred

    def obtain_correct_classified_instances(self, y, t, epoch, train, val, test, batch_size):
        """Obtain the number of correctly classified instances by taking the mean prediction over all
        sub sequences and compare it against the true label.

        Parameters
        ----------
        y : torch.Tensor
            Tensor containing the true labels. Shape [batch_size, output_size].
        t : int
            The time step in the data loader.
        epoch : int
            The epoch during training.
        train : Boolean
            If in train mode.
        val : Boolean
            If in validation mode.
        test : Boolean
            If in test mode.
        batch_size : int
            The number of samples in the batch.
        Returns
        -------
        num_correct : int
            Percentage of change in prediction within the total sequences.
        """
        assert super().check_if_only_one_boolean(train, val,
                                                 test), f"You can only be in either one of train/val/test mode, " \
                                                        f"you are now in " \
                                                        f"{len([var for var in [train, val, test] if var == True])} " \
                                                        f"modes"
        if train:
            predictions = self.train_predictions
        elif val:
            predictions = self.val_predictions
        elif test:
            predictions = self.test_predictions

        assert len(predictions) == self.num_sub_sequences, f'The number of predictions on time step {t} ' \
                                                           f'should be {self.num_sub_sequences}, but ' \
                                                           f'is {len(predictions)}'

        perc_changes_in_pred, mean_pred = self.analyze_predictions(train, val, test, batch_size)
        print(
            f'The percentage of change in prediction within the entire sequence is: {perc_changes_in_pred:.2f}% in '
            f'epoch {epoch} and batch {t}'
        )

        # y gets reshaped into the same form as pred in order to compute the accuracy
        true_labels = list(np.array(y.view(1, -1), dtype=np.int64).flat)
        mean_pred = list(np.array(mean_pred, dtype=np.int64).flat)

        num_correct = len([i for i, j in zip(true_labels, mean_pred) if i == j])

        return num_correct, mean_pred

    def evaluate_after_entire_sequence(self, y, t, epoch, train, val, test, batch_size):
        """
        After an entire sequence is processed, we want to know the average prediction over the entire
        sequence and use that as a final prediction for the entire sequence.
        """
        assert super().check_if_only_one_boolean(train, val,
                                                 test), f"You can only be in either one of train/val/test mode, " \
                                                        f"you are now in " \
                                                        f"{len([var for var in [train, val, test] if var == True])} " \
                                                        f"modes"

        # Data is first moved to cpu and then converted to numpy array
        true_label = y.cpu().data.numpy()
        correct_entire_seq, mean_pred = self.obtain_correct_classified_instances(y, t, epoch,
                                                                                 train=train,
                                                                                 val=val,
                                                                                 test=test,
                                                                                 batch_size=batch_size)

        # We clear the list with the predictions for the next batch which contains the sequences of new
        # rec ids. We need to add NaN values for the padded parts of the sequence. These NaN values
        # will be disregarded when taking the mean over all sub-sequences.
        if train:
            self.final_train_predictions.append(mean_pred)
            self.train_predictions.clear()
            self.train_probabilities = self.add_nans_for_padded_sequences(self.train_probabilities, batch_size)
            mean_prob = np.nanmean(np.stack(self.train_probabilities, axis=0), axis=0)
            self.final_train_prob.append(mean_prob)
            self.train_probabilities.clear()
        elif val:
            self.final_val_predictions.append(mean_pred)
            self.val_predictions.clear()
            self.val_probabilities = self.add_nans_for_padded_sequences(self.val_probabilities, batch_size)
            mean_prob = np.nanmean(np.stack(self.val_probabilities, axis=0), axis=0)
            self.final_val_prob.append(mean_prob)
            self.val_probabilities.clear()
        elif test:
            self.test_labels.append(true_label)
            self.final_test_predictions.append(mean_pred)
            self.test_predictions.clear()
            self.test_probabilities = self.add_nans_for_padded_sequences(self.test_probabilities, batch_size)
            mean_prob = np.nanmean(np.stack(self.test_probabilities, axis=0), axis=0)
            self.final_test_prob.append(mean_prob)
            self.test_probabilities.clear()

        return correct_entire_seq

    def calculate_masked_loss_and_pred(self, y, y_pred, lengths):
        """Calculate the masked loss and predictions of y and y_pred based on the lengths.

        If an item in variable 'lengths' has length zero, then it means that for that sample
        only padded entries are present. Thus it is an empty sequence. We will therefore mask
        those samples when computing the loss and also mask the prediction as that prediction
        is useless.

        Parameters
        ----------
        y : torch.Tensor
            Tensor containing the true labels. Shape [batch_size, output_size].
        y_pred : torch.Tensor
            Tensor containing the predictions/logits. Shape [batch_size, output_size].
        lengths : torch.Tensor
            Tensor containing the sequence length of each sample in the batch.

        Returns
        -------
        loss : torch.Tensor
            The mean loss over all samples in the batch that have a sequence length > 0.
        y_pred[mask_padded] : torch.Tensor
            The masked predictions/logits. Shape [len(non masked samples), output_size].
        """
        # Mask all predictions of the samples in the batch for which the entire sequence has
        # already been processed. If the entire sequence already has been processed, it means
        # that the consecutive part of that sequence only consists of padded items.
        lengths_reshaped = lengths.view(y_pred.shape[0], y_pred.shape[1])

        mask_padded = lengths_reshaped > 0

        # Computes loss (input for BCEwithlogits loss should be (logits, true label))
        # loss_list is a list with the unreduced loss for each sample in the batch, but with the
        # padded samples masked. Therefore, the padded samples do not contribute to the loss in any way
        loss_list = self.loss_fn(y_pred[mask_padded], y[mask_padded])

        # Calculate the mean loss over all non-padded samples in the batch
        loss = sum(loss_list) / sum(mask_padded)

        return loss, y_pred[mask_padded]

    def train_step(self, x, y, x_lengths):
        """Returns masked loss and prediction."""
        # Sets model to train mode
        self.model.train()

        # Zeroes gradients
        self.optimizer.zero_grad()

        # Forward pass to get output/logits. For BCEwithlogits loss, y_pred will be logits
        y_pred = self.model(x, x_lengths)

        loss, y_pred_masked = self.calculate_masked_loss_and_pred(y, y_pred, x_lengths)

        # We only perform the backward pass and update the parameters if the loss contains non nan values.
        # This means that batches that contain solely samples with only padded values, the backward
        # pass and parameter update will not be performed (as the entire sequence already has been processed
        # and there is nothing to update).
        if not math.isnan(loss.item()):
            # Getting gradients w.r.t. parameters
            # As long as the padded entries do not contribute to the loss, their gradient will always be zero
            loss.backward()

            # Updates parameters
            self.optimizer.step()

        return loss, y_pred_masked

    def train(self, train_loader, val_loader, train_lengths, val_lengths, n_epochs=50):
        """The model is trained/validated/tested according to a stateful LSTM model. Meaning that
        we will propagate hidden states and cells up until an entire sequence is processed. The
        entire sequence length is 28800 time steps and the sub_sequence length is 200 time steps. This
        means that 1 batch contains 200 time steps of batch_size (60) rec ids. As a result, we will
        reset the hidden states and cells after 144 batches have been processed. Then we will start
        processing the entire sequences of the next batch_size rec ids and again reset the hidden states
        and cells after 144 batches have been processed. This process is repeated up until the entire
        sequence of each rec id has been processed.

        As the sequence of each rec id has been truncated/padded to the same length, we will mask
        the padded data points such that it does not contribute to the loss function or predictions.
        For the sequences that have been padded with more than sub_sequence length (=200) time steps,
        this means that the last batches for that rec id contains solely padded values.

        Parameters
        ----------
        train_loader : DataLoader
            Must be of shape [batch_size, sequence_length, num_features].
        val_loader : DataLoader
            Must be of shape [batch_size, sequence_length, num_features].
        train_lengths : torch.LongTensor
            Tensor containing the unpadded sequence length of each sample for each train batch.
            Will be used to mask the padded data points in the sequence in the loss.
        val_lengths : torch.LongTensor
            Tensor containing the unpadded sequence length of each sample for each val batch.
            Will be used to mask the padded data points in the sequence in the loss.
        """
        output_path = file_paths['output_path'] + "/" + "model"

        current_date_and_time = "{:%Y-%m-%d_%H-%M}".format(datetime.datetime.now())
        file_name_output = f'{current_date_and_time}_best_model_stateful'

        for epoch in range(1, n_epochs + 1):
            total_train_loss_epoch = 0.0
            correct_train, total_samples_train = 0, 0
            for t, (x_batch, y_batch) in enumerate(train_loader):

                # The order in the batch MUST be [batch_size, sequence length, num_features]
                batch_size_train = x_batch.shape[0]

                # This variable contains the 'true' subsequence length of each sample in the batch.
                # Meaning that it contains the number of the original time steps (so not padded)
                # that are present for each sample in the batch
                batch_lengths_train = train_lengths[t * batch_size_train:(t * batch_size_train + batch_size_train)]

                # If there is a sample in the batch that contains of solely padded values (time steps),
                # then that sample will be disregarded when calculating the loss and making predictions.
                true_batch_size_train = sum(batch_lengths_train > 0)
                total_samples_train += true_batch_size_train

                # We manually reset the hidden states and cells after an entire sequence is processed
                if t % self.num_sub_sequences == 0:
                    print(
                        f'The hidden states and cells are resetted at the beginning of epoch {epoch}, '
                        f'batch {t} of train_loader')
                    self.reset_states()

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # avg_train_batch_loss contains the value of the average cost of all the training examples of
                # the current batch
                # y_pred is with the masked predictions
                avg_train_batch_loss, y_pred = self.train_step(x_batch, y_batch, batch_lengths_train)

                super().get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                # After an entire sequence is processed, we want to know the average prediction over the entire
                # sequence and use that as a final prediction for the entire sequence
                if len(self.train_predictions) == self.num_sub_sequences:
                    correct_train_entire_seq = self.evaluate_after_entire_sequence(y_batch, t, epoch,
                                                                                   train=True, val=False,
                                                                                   test=False,
                                                                                   batch_size=batch_size_train)

                    correct_train += correct_train_entire_seq

                # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                total_train_loss_epoch += (np.nan_to_num(avg_train_batch_loss.item()) * true_batch_size_train)

                # We do truncated BPTT as the model otherwise would have to backpropagate through the entire
                # sequence, which will lead to very long training times + vanishing/exploding gradients
                self.model.lstm._hidden_state = self.model.lstm._hidden_state.detach()
                self.model.lstm._hidden_cell = self.model.lstm._hidden_cell.detach()

            # Calculate average sample train loss for this particular epoch.
            avg_sample_train_loss_epoch = total_train_loss_epoch / total_samples_train
            self.train_losses.append(avg_sample_train_loss_epoch)

            # We clear the labels and probabilities after each epoch
            self.final_train_prob.clear()

            self.model.eval()
            with torch.no_grad():
                total_val_loss_epoch = 0.0
                correct_val, total_samples_val = 0, 0
                for t, (x_val, y_val) in enumerate(val_loader):
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    if t % self.num_sub_sequences == 0:
                        print(f'The hidden states and cells are resetted at the beginning of epoch {epoch}, '
                              f'batch {t} of val_loader')
                        self.reset_states()

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size_val = x_val.shape[0]

                    # This variable contains the 'true' subsequence length of each sample in the batch.
                    # Meaning that it contains the number of the original time steps (so not padded)
                    # that are present for each sample in the batch
                    batch_lengths_val = val_lengths[t * batch_size_val:(t * batch_size_val + batch_size_val)]

                    # If there is a sample in the batch that contains of solely padded values (time steps),
                    # then that sample will be disregarded when calculating the loss and making predictions.
                    true_batch_size_val = sum(batch_lengths_val > 0)
                    total_samples_val += true_batch_size_val

                    # forward pass: compute predicted outputs by passing inputs to the model
                    # output shape is [batch_size, output_dim] and contains logits
                    y_pred = self.model(x_val, batch_lengths_val)
                    avg_val_batch_loss, y_pred_masked = self.calculate_masked_loss_and_pred(y_val, y_pred,
                                                                                            batch_lengths_val)

                    super().get_predictions_binary_model(y_pred_masked, train=False, val=True, test=False)

                    if len(self.val_predictions) == self.num_sub_sequences:
                        correct_val_entire_seq = self.evaluate_after_entire_sequence(y_val, t, epoch,
                                                                                     train=False,
                                                                                     val=True,
                                                                                     test=False,
                                                                                     batch_size=batch_size_val)

                        correct_val += correct_val_entire_seq

                    # total_val_loss_epoch accumulates the total loss per validation batch for the entire epoch
                    # update running training loss
                    total_val_loss_epoch += (np.nan_to_num(avg_val_batch_loss.item()) * true_batch_size_val)

                # Calculate average sample validation loss for this particular epoch.
                avg_sample_val_loss_epoch = total_val_loss_epoch / total_samples_val
                self.val_losses.append(avg_sample_val_loss_epoch)

                # We clear the probabilities after each epoch
                self.final_val_prob.clear()

            print(
                f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.4f}\t "
                f"Validation loss: {avg_sample_val_loss_epoch:.4f}"
                )
            print('-' * 50)

        print('TRAINING COMPLETE')

        return avg_sample_val_loss_epoch

    def final_train(self, train_val_loader, train_val_lengths, n_epochs=50):
        """The model is trained according to a stateful LSTM model. Meaning that
        we will propagate hidden states and cells up until an entire sequence is processed. The
        entire sequence length is 28800 time steps and the sub_sequence length is 200 time steps. This
        means that 1 batch contains 200 time steps of batch_size (60) rec ids. As a result, we will
        reset the hidden states and cells after 144 batches have been processed. Then we will start
        processing the entire sequences of the next batch_size rec ids and again reset the hidden states
        and cells after 144 batches have been processed. This process is repeated up until the entire
        sequence of each rec id has been processed.

        As the sequence of each rec id has been truncated/padded to the same length, we will mask
        the padded data points such that it does not contribute to the loss function or predictions.
        For the sequences that have been padded with more than sub_sequence length (=200) time steps,
        this means that the last batches for that rec id contains solely padded values.

        Parameters
        ----------
        train_val_loader : DataLoader
            Concatenated DataLoader of train and val test. Datasets are concatenated using ConcatDataset.
            Must be of shape [batch_size, sequence_length, num_features].
        train_val_lengths : torch.LongTensor
            Tensor containing the unpadded sequence length of each sample for each batch.
            Will be used to mask the padded data points in the sequence in the loss.
        n_epochs : int
            Number of epochs to train the model.
        """
        output_path = file_paths['output_path'] + "/" + "model"

        current_date_and_time = "{:%Y-%m-%d_%H-%M}".format(datetime.datetime.now())
        file_name_output = f'{current_date_and_time}_best_model_statefull_final_train'

        for epoch in range(1, n_epochs + 1):
            total_train_loss_epoch = 0.0
            correct_train, total_samples_train = 0, 0
            for t, (x_batch, y_batch) in enumerate(train_val_loader):

                # The order in the batch MUST be [batch_size, sequence length, num_features]
                batch_size = x_batch.shape[0]

                # This variable contains the 'true' subsequence length of each sample in the batch.
                # Meaning that it contains the number of the original time steps (so not padded)
                # that are present for each sample in the batch
                batch_lengths = train_val_lengths[t * batch_size:(t * batch_size + batch_size)]

                # If there is a sample in the batch that contains of solely padded values (time steps),
                # then that sample will be disregarded when calculating the loss and making predictions.
                true_batch_size = sum(batch_lengths > 0)
                total_samples_train += true_batch_size

                # We manually reset the hidden states and cells after an entire sequence is processed
                if t % self.num_sub_sequences == 0:
                    print(f'The hidden states and cells are resetted at the beginning of epoch {epoch}, '
                          f'batch {t} of train_loader')
                    self.reset_states()

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # avg_train_batch_loss contains the value of the average cost of all the training examples of
                # the current batch
                # y_pred is with the masked predictions
                avg_train_batch_loss, y_pred = self.train_step(x_batch, y_batch, batch_lengths)

                super().get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                # After an entire sequence is processed, we want to know the average prediction over the entire
                # sequence and use that as a final prediction for the entire sequence
                if len(self.train_predictions) == self.num_sub_sequences:
                    correct_train_entire_seq = self.evaluate_after_entire_sequence(y_batch, t, epoch,
                                                                                   train=True, val=False,
                                                                                   test=False, batch_size=batch_size)

                    correct_train += correct_train_entire_seq

                # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                total_train_loss_epoch += (np.nan_to_num(avg_train_batch_loss.item()) * true_batch_size)

                # We do truncated BPTT as the model otherwise would have to backpropagate through the entire
                # sequence, which will lead to very long training times + vanishing/exploding gradients
                self.model.lstm.hidden_state = self.model.lstm.hidden_state.detach()
                self.model.lstm.hidden_cell = self.model.lstm.hidden_cell.detach()

            # Calculate average sample train loss for this particular epoch.
            avg_sample_train_loss_epoch = total_train_loss_epoch / total_samples_train
            self.train_losses.append(avg_sample_train_loss_epoch)

            # We clear the labels and probabilities after each epoch
            self.final_train_prob.clear()

            print(f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.10f}")
            print('-' * 50)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion, }, f'{output_path}/{file_name_output}.pth')
        print('TRAINING COMPLETE')

    def evaluate(self, test_loader, test_lengths, checkpoint):
        """Evaluate the model by making predictions over each subsequence and after all
        subsequences for each rec id have been processed take the mean over all subsequence predictions."""

        model_epoch = checkpoint['epoch']
        print(f"Model was saved at {model_epoch} epochs\n")
        print(f'Loading at epoch {model_epoch} saved model weights...')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        with torch.no_grad():
            correct_test, total_samples_test = 0, 0
            for t, (x_test, y_test) in enumerate(test_loader):
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)

                # The order in the batch MUST be [batch_size, sequence length, num_features]
                batch_size_test = x_test.shape[0]

                batch_lengths_test = test_lengths[t * batch_size_test:(t * batch_size_test + batch_size_test)]

                # If there is a sample in the batch that contains of solely padded values (time steps),
                # then that sample will be disregarded when calculating the loss and making predictions.
                true_batch_size_test = sum(batch_lengths_test > 0)
                total_samples_test += true_batch_size_test

                # We manually reset the hidden states and cells after an entire sequence is processed
                if t % self.num_sub_sequences == 0:
                    print(f'The hidden states and cells are resetted at the beginning batch {t} of test_loader')
                    self.reset_states()

                # forward pass: compute predicted outputs by passing inputs to the model
                # output shape is [batch_size, 1] and contains logits
                y_pred = self.model(x_test, batch_lengths_test)

                _, y_pred_masked = self.calculate_masked_loss_and_pred(y_test, y_pred, batch_lengths_test)

                super().get_predictions_binary_model(y_pred_masked, train=False, val=False, test=True)

                # After the entire sequence has been processed we take the mean prediction of all sub-sequence
                # predictions
                if len(self.test_predictions) == self.num_sub_sequences:
                    true_label = y_test.cpu().data.numpy()
                    correct_test_entire_seq = self.evaluate_after_entire_sequence(y_test, t, 0,
                                                                                  train=False, val=False,
                                                                                  test=True,
                                                                                  batch_size=batch_size_test)

                    correct_test += correct_test_entire_seq

                    self.final_test_predictions = list(np.array(self.final_test_predictions, dtype=np.int64).flat)
                    self.final_test_prob = list(np.array(self.final_test_prob, dtype=np.int64).flat)

                    num_correct_batch = len([i for i, j in zip(self.final_test_predictions, true_label) if i == j])
                    print(f'The number of correct predicted samples: '
                          f'{num_correct_batch}/{len(self.final_test_predictions)}')

                    precision_test, recall_test, _ = precision_recall_curve(true_label, self.final_test_prob)
                    super().plot_prec_recall_curve(recall_test, precision_test)

                    print(f'Precision score: {precision_score(true_label, self.final_test_predictions)}')
                    print(f'Recall score: {recall_score(true_label, self.final_test_predictions)}')
                    print(f'F1 score: {f1_score(true_label, self.final_test_predictions)}')
                    print(f'AUC score: {roc_auc_score(true_label, list(np.array(self.final_test_prob).flat))}')


class OptimizationStatefulFeatureSequenceLSTM(OptimizationLSTM):
    """Train, optimize and evaluate a stateful LSTM model. Stateful means that the hidden
    states and cells are passed on to the next batch instead of being resetted to zero at
    the beginning of each batch. This is useful when the sequence is too long to be processed
    by the LSTM model all at once. We therefore split up the sequence in multiple sub-sequences.
    The number of sub-sequences needed to process an entire sequence is denoted by the parameter
    num_sub_sequences. The hidden cells and states are manually resetted to zero after num_sub_sequences
    batches have been processed.

    Each sequence has been padded or truncated to the same length. We will mask the padded data points
    such that it does not contribute to the loss function.

    The final prediction will be made by making a prediction for each sub-sequence and at the end take
    the mean over all sub-sequence predictions.

    Parameters
    ----------
    model : nn.Module
        LSTM model with specified input_dim, hidden_dim, output_dim, batch_size, device,
        layer_dim, batch_first.
    loss_fn : torch.nn.modules.loss
        Loss function
    optimizer : torch.optim
        Optimizer function.
    num_sub_sequences : int
        Number of sub-sequences that make up an entire sequence.
    """
    def __init__(self, model, loss_fn, optimizer, device, num_sub_sequences):
        super().__init__(model, loss_fn, optimizer, device)
        self.num_sub_sequences = num_sub_sequences
        self.train_predictions = []
        self.val_predictions = []
        self.test_predictions = []
        self.train_probabilities = []
        self.val_probabilities = []
        self.test_probabilities = []
        self.final_train_predictions = []
        self.final_val_predictions = []
        self.final_test_predictions = []
        self.final_train_prob = []
        self.final_val_prob = []
        self.final_test_prob = []

        self.final_max_val_probabilities = []
        self.final_max_test_predictions = []
        self.final_max_test_probabilities = []
        self.train_losses = []
        self.val_losses = []
        self.val_labels = []
        self.test_labels = []

    def reset_states(self, batch_size):
        """Reset the hidden states and cells to zero."""
        for layer in self.model.modules():
            if hasattr(layer, 'reset_hidden_cell'):
                layer.reset_hidden_cell(batch_size)
            if hasattr(layer, 'reset_hidden_state'):
                layer.reset_hidden_state(batch_size)

    def get_mean_prediction_and_probability(self, train, val, test):
        """Get the mean pred and prob over all sub-sequences.

        If a total sequence is 50 time steps and the sub_sequence length is 10 time steps,
        then there are 50 / 10 = 5 sub_sequences. Meaning, that we'll take the mean over the
        5 sub-sequences to obtain the mean pred and prob.

        Returns
        -------
        mean_pred : List[int]
            List of size [batch_size]. Containing the mean prediction for all batch_size rec ids.
        mean_prob : List[float]
            List of size [batch_size]. Containing the mean probability for all batch_size rec ids.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, you " \
                                                     f"are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        if train:
            predictions = self.train_predictions
            probabilities = self.train_probabilities
        elif val:
            predictions = self.val_predictions
            probabilities = self.val_probabilities
        elif test:
            predictions = self.test_predictions
            probabilities = self.test_probabilities

        # Calculate the mean prediction by taking the average over all sub_sequences that make up
        # one total sequence. The variable predictions contains all the sub_sequence predictions
        # belonging to batch_size rec ids.

        mean_pred = list(np.array(np.mean(np.stack(predictions, axis=1), axis=1), dtype=np.int64).flat)
        mean_prob = list(np.array(np.mean(np.stack(probabilities, axis=1), axis=1)).flat)

        max_pred = list(np.array(np.max(np.stack(predictions, axis=1), axis=1)).flat)
        max_prob = list(np.array(np.max(np.stack(probabilities, axis=1), axis=1)).flat)

        return mean_pred, mean_prob, max_pred, max_prob

    def obtain_correct_classified_instances(self, t, train, val, test):
        """Obtain the number of correctly classified instances by taking the mean prediction over all
        sub sequences and compare it against the true label.

        Parameters
        ----------
        y : torch.Tensor
            Tensor containing the true labels. Shape [batch_size, output_size].
        t : int
            The time step in the data loader.
        train : Boolean
            If in train mode.
        val : Boolean
            If in validation mode.
        test : Boolean
            If in test mode.
        Returns
        -------
        num_correct : int
            Percentage of change in prediction within the total sequences.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, " \
                                                     f"you are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        if train:
            predictions = self.train_predictions
        elif val:
            predictions = self.val_predictions
        elif test:
            predictions = self.test_predictions

        assert len(predictions) == self.num_sub_sequences, f'The number of predictions on time step {t} ' \
                                                           f'should be {self.num_sub_sequences}, but ' \
                                                           f'is {len(predictions)}'

        mean_pred, mean_prob, max_pred, max_prob = self.get_mean_prediction_and_probability(train, val, test)

        return mean_pred, mean_prob, max_pred, max_prob

    def evaluate_after_entire_sequence(self, y, t, train, val, test):
        """
        After an entire sequence is processed, we want to know the average prediction over the entire
        sequence and use that as a final prediction for the entire sequence.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, " \
                                                     f"you are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"

        # Data is first moved to cpu and then converted to numpy array
        true_label = y.cpu().data.numpy()
        mean_pred, mean_prob, max_pred, max_prob = self.obtain_correct_classified_instances(t,
                                                                                            train=train,
                                                                                            val=val,
                                                                                            test=test)

        # We clear the list with the predictions for the next batch which contains the sequences of new
        # rec ids
        if train:
            self.final_train_predictions.append(mean_pred)
            self.train_predictions.clear()
            self.final_train_prob.append(mean_prob)
            self.train_probabilities.clear()

        elif val:
            self.val_labels.append(true_label)
            self.final_val_predictions.append(mean_pred)
            self.val_predictions.clear()
            self.final_val_prob.append(mean_prob)
            self.val_probabilities.clear()
            self.final_max_val_probabilities.append(max_prob)

        elif test:
            self.test_labels.append(true_label)
            self.final_test_predictions.append(mean_pred)
            self.test_predictions.clear()
            self.final_test_prob.append(mean_prob)
            self.test_probabilities.clear()
            self.final_max_test_predictions.append(max_pred)
            self.final_max_test_probabilities.append(max_prob)

        return mean_pred, mean_prob, max_pred, max_prob

    def train_step(self, x, y):
        """Returns loss and prediction."""
        # Sets model to train mode
        self.model.train()

        # Zeroes gradients
        self.optimizer.zero_grad()

        # Forward pass to get output/logits. For BCEwithlogits loss, y_pred will be logits
        y_pred = self.model(x)

        loss = self.loss_fn(y_pred, y)

        if loss.isnan().any():
            print(f'Loss before backward pass: {loss}')

        # Getting gradients w.r.t. parameters
        loss.backward()

        if loss.isnan().any():
            print(f'Loss after backward pass: {loss}')

        # Updates parameters
        self.optimizer.step()

        return loss, y_pred

    def train(self, trial, train_loader_list: List[DataLoader], test_loader_list: List[DataLoader],
              params, n_epochs=50):
        """The model is trained/validated/tested on each sub-sequence up until each entire
        sequence has been processed. The entire sequence length can be 50 time steps and the
        sub_sequence length can be 10 time steps for instance. We will make a prediction after each sub-sequence
        has been processed.

        This means that 1 batch contains 10 time steps of batch_size rec ids. As a result, we will
        reset all predictions after 5 batches have been processed. Then we will start processing
        the entire sequences of the next batch_size rec ids and again reset after 5 batches have
        been processed. This process is repeated up until the entire sequence of each rec id has
        been processed.

        Parameters
        ----------
        trial :
        train_loader_list : List[DataLoader]
            List containing one or more DataLoader objects containing training data.
        test_loader_list : List[DataLoader]
            List containing one or more DataLoader objects containing test data.
        params : dict
            Dictionary with the hyperparameters and its values.
        n_epochs : int
            Number of epochs for which you want to train your model.
        """
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(1, n_epochs + 1):
            total_train_loss_epoch = 0.0
            correct_train, total_samples_train = 0, 0
            # Clear at the beginning of every epoch
            self.final_train_prob.clear()
            self.final_val_prob.clear()
            self.val_labels.clear()

            self.train_losses.clear()
            self.val_losses.clear()

            self.final_val_predictions.clear()
            self.final_max_val_probabilities.clear()

            for train_loader in train_loader_list:
                for t, (x_batch, y_batch) in enumerate(train_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size_train = x_batch.shape[0]

                    total_samples_train += batch_size_train

                    # We manually reset the hidden states and cells after an entire sequence is processed
                    if t % self.num_sub_sequences == 0:
                        print(f'The hidden states and cells are resetted at the beginning of epoch {epoch}, '
                              f'batch {t} of train_loader')
                        self.reset_states(batch_size=batch_size_train)

                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # avg_train_batch_loss contains the value of the average cost of all the training examples of
                    # the current batch
                    # y_pred is with the masked predictions
                    avg_train_batch_loss, y_pred = self.train_step(x_batch, y_batch)

                    super().get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                    # After an entire sequence is processed, we clear the predictions and probs
                    if len(self.train_predictions) == self.num_sub_sequences:
                        self.train_predictions.clear()
                        self.train_probabilities.clear()

                    # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                    total_train_loss_epoch += (np.nan_to_num(avg_train_batch_loss.item()) * batch_size_train)

                    # We do truncated BPTT as the model otherwise would have to backpropagate through the entire
                    # sequence, which will lead to very long training times + vanishing/exploding gradients
                    self.model.lstm.hidden_state = self.model.lstm.hidden_state.detach()
                    self.model.lstm.hidden_cell = self.model.lstm.hidden_cell.detach()

                # Calculate average sample train loss for this particular epoch.
                avg_sample_train_loss_epoch = total_train_loss_epoch / total_samples_train
                self.train_losses.append(avg_sample_train_loss_epoch)

            self.model.eval()
            torch.autograd.set_detect_anomaly(True)
            with torch.no_grad():
                total_val_loss_epoch = 0.0
                correct_val, total_samples_val = 0, 0
                all_val_probabilities = []

                for val_loader in test_loader_list:
                    for t, (x_val, y_val) in enumerate(val_loader):

                        # The order in the batch MUST be [batch_size, sequence length, num_features]
                        batch_size_val = x_val.shape[0]

                        # We manually reset the hidden states and cells after an entire sequence is processed
                        if t % self.num_sub_sequences == 0:
                            print(f'The hidden states and cells are resetted at the beginning of epoch {epoch}, '
                                  f'batch {t} of val_loader')
                            self.reset_states(batch_size=batch_size_val)

                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)

                        total_samples_val += batch_size_val

                        # forward pass: compute predicted outputs by passing inputs to the model
                        # output shape is [batch_size, output_dim] and contains logits
                        y_pred = self.model(x_val)

                        avg_val_batch_loss = self.loss_fn(y_pred, y_val)

                        self.get_predictions_binary_model(y_pred, train=False, val=True, test=False)

                        if len(self.val_predictions) == self.num_sub_sequences:
                            all_val_probabilities.append(self.val_probabilities.copy())
                            mean_pred, mean_prob, max_pred, max_prob = self.evaluate_after_entire_sequence(y_val, t,
                                                                                                           train=False,
                                                                                                           val=True,
                                                                                                           test=False)
                            # self.val_predictions.clear()
                            # self.val_probabilities.clear()

                        # total_val_loss_epoch accumulates the total loss per validation batch for the entire epoch
                        total_val_loss_epoch += (avg_val_batch_loss.item() * batch_size_val)

                # Calculate average sample validation loss for this particular epoch
                avg_sample_val_loss_epoch = total_val_loss_epoch / total_samples_val
                self.val_losses.append(avg_sample_val_loss_epoch)

                val_labels_final = list(np.array(np.concatenate([array for array in self.val_labels], axis=0),
                                                dtype=np.int64).flat)
                final_val_prob_final = list(np.array(np.concatenate([array for array in self.final_val_prob],
                                                                   axis=0)).flat)

                final_max_val_probabilities_final = list(
                    np.array(np.concatenate([array for array in self.final_max_val_probabilities], axis=0)).flat)

                auc_mean_val = roc_auc_score(val_labels_final, final_val_prob_final)
                auc_max_val = roc_auc_score(val_labels_final, final_max_val_probabilities_final)

            print(f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.4f}\t "
                  f"Validation loss: {avg_sample_val_loss_epoch:.4f} \t AUC mean val: {auc_mean_val} \t AUC max val: {auc_max_val}")
            print('-' * 50)

        print('TRAINING COMPLETE')

        return auc_mean_val, auc_max_val, params

    def final_train(self, train_loader_list, feature_name, add_static_data, fold_i, n_epochs=50):
        """The final model will be trained with the optimal hyperparameters on the train+val datset."""
        output_path = file_paths['output_path'] + "/" + "model"

        current_date_and_time = "{:%Y-%m-%d_%H-%M}".format(datetime.datetime.now())

        if add_static_data:
            file_name_output = f'{current_date_and_time}_fold_{fold_i}_best_model_lstm_feature_{feature_name}_combined_seq_final_train'

        if not add_static_data:
            file_name_output = f'{current_date_and_time}_fold_{fold_i}_best_model_lstm_feature_{feature_name}_seq_final_train'

        for epoch in range(1, n_epochs + 1):
            self.final_train_prob.clear()
            total_train_loss_epoch = 0.0
            correct_train, total_samples_train = 0, 0

            for train_loader in train_loader_list:
                for t, (x_batch, y_batch) in enumerate(train_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size = x_batch.shape[0]
                    total_samples_train += batch_size

                    # We manually reset the hidden states and cells after an entire sequence is processed
                    if t % self.num_sub_sequences == 0:
                        print(f'The hidden states and cells are resetted at the beginning of epoch {epoch}, '
                              f'batch {t} of train_loader')
                        self.reset_states(batch_size=batch_size)

                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # avg_train_batch_loss contains the value of the average cost of all the training examples of
                    # the current batch
                    # y_pred is with the masked predictions
                    avg_train_batch_loss, y_pred = self.train_step(x_batch, y_batch)

                    self.get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                    # After an entire sequence is processed, we want to know the average prediction over the entire
                    # sequence and use that as a final prediction for the entire sequence
                    if len(self.train_predictions) == self.num_sub_sequences:
                        self.train_predictions.clear()
                        self.train_probabilities.clear()

                    # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                    total_train_loss_epoch += (avg_train_batch_loss.item() * batch_size)

                    # We do truncated BPTT as the model otherwise would have to backpropagate through the entire
                    # sequence, which will lead to very long training times + vanishing/exploding gradients
                    self.model.lstm.hidden_state = self.model.lstm.hidden_state.detach()
                    self.model.lstm.hidden_cell = self.model.lstm.hidden_cell.detach()

            # Calculate average sample train loss for this particular epoch.
            avg_sample_train_loss_epoch = total_train_loss_epoch / total_samples_train
            self.train_losses.append(avg_sample_train_loss_epoch)

            print(f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.10f}")
            print('-' * 50)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn}, f'{output_path}/{file_name_output}.pth')
        print('TRAINING COMPLETE')
        print(f'Model is saved at: {output_path}/{file_name_output}.pth')

    def evaluate(self, test_loader, checkpoint):
        """Evaluate the model by making predictions over each subsequence and after all
        subsequences for each rec id have been processed take the mean over all subsequence predictions."""
        all_test_predictions = []
        all_test_probabilities = []

        model_epoch = checkpoint['epoch']
        print(f"Model was saved at {model_epoch} epochs\n")
        print(f'Loading at epoch {model_epoch} saved model weights...')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        with torch.no_grad():
            correct_test, total_samples_test = 0, 0
            for data_loader in test_loader:
                for t, (x_test, y_test) in enumerate(data_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size_test = x_test.shape[0]

                    x_test = x_test.to(self.device)
                    y_test = y_test.to(self.device)

                    total_samples_test += batch_size_test

                    # We manually reset the hidden states and cells after an entire sequence is processed
                    if t % self.num_sub_sequences == 0:
                        print(f'The hidden states and cells are resetted at the beginning batch {t} of test_loader')
                        self.reset_states(batch_size=batch_size_test)

                    # forward pass: compute predicted outputs by passing inputs to the model
                    # output shape is [batch_size, 1] and contains logits
                    y_pred = self.model(x_test)

                    self.get_predictions_binary_model(y_pred, train=False, val=False, test=True)

                    # After the entire sequence has been processed we take the mean prediction of all sub-sequence
                    # predictions
                    if len(self.test_predictions) == self.num_sub_sequences:
                        all_test_predictions.append(self.test_predictions.copy())
                        all_test_probabilities.append(self.test_probabilities.copy())
                        mean_pred, mean_prob, max_pred, max_prob = self.evaluate_after_entire_sequence(y_test, t,
                                                                                                       train=False,
                                                                                                       val=False,
                                                                                                       test=True
                                                                                                       )

        self.test_labels = list(np.array(np.concatenate([array for array in self.test_labels], axis=0),
                                         dtype=np.int64).flat)

        self.final_test_prob = list(np.array(np.concatenate([array for array in self.final_test_prob], axis=0)).flat)
        self.final_test_predictions = list(np.array(np.concatenate([array for array in self.final_test_predictions],
                                                                   axis=0)).flat)

        self.final_max_test_predictions = list(
            np.array(np.concatenate([array for array in self.final_max_test_predictions], axis=0)).flat)
        self.final_max_test_probabilities = list(
            np.array(np.concatenate([array for array in self.final_max_test_probabilities], axis=0)).flat)

        print(f'Precision score with mean pred: {precision_score(self.test_labels, self.final_test_predictions)}')
        print(f'Recall score with mean pred: {recall_score(self.test_labels, self.final_test_predictions)}')
        print(f'Specificity score with mean pred: {recall_score(np.logical_not(self.test_labels), np.logical_not(self.final_test_predictions))}')
        print(f'F1 score with mean pred: {f1_score(self.test_labels, self.final_test_predictions)}')
        print(f'AUC score with mean prob: {roc_auc_score(self.test_labels, self.final_test_prob)}')
        print(f'AP score with mean prob: {average_precision_score(self.test_labels, self.final_test_prob)}')

        print(f'Precision score with max pred: {precision_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'Recall score with max pred: {recall_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'Specificity score with max pred: {recall_score(np.logical_not(self.test_labels), np.logical_not(self.final_max_test_predictions))}')
        print(f'F1 score with max pred: {f1_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'AUC score with max prob: {roc_auc_score(self.test_labels, self.final_max_test_probabilities)}')
        print(f'AP score with max prob: {average_precision_score(self.test_labels, self.final_max_test_probabilities)}')

        results_dict = {f'precision_mean_pred': precision_score(self.test_labels, self.final_test_predictions),
                        f'recall_mean_pred': recall_score(self.test_labels, self.final_test_predictions),
                        f'specificity_mean_pred': recall_score(np.logical_not(self.test_labels),
                                                               np.logical_not(self.final_test_predictions)),
                        f'f1_mean_pred': f1_score(self.test_labels, self.final_test_predictions),
                        f'auc_mean_prob': roc_auc_score(self.test_labels, self.final_test_prob),
                        f'ap_mean_prob': average_precision_score(self.test_labels, self.final_test_prob),
                        f'precision_max_pred': precision_score(self.test_labels, self.final_max_test_predictions),
                        f'recall_max_pred': recall_score(self.test_labels, self.final_max_test_predictions),
                        f'specificity_max_pred': recall_score(np.logical_not(self.test_labels),
                                                              np.logical_not(self.final_max_test_predictions)),
                        f'f1_max_pred': f1_score(self.test_labels, self.final_max_test_predictions),
                        f'auc_max_prob': roc_auc_score(self.test_labels, self.final_max_test_probabilities),
                        f'ap_max_prob': average_precision_score(self.test_labels, self.final_max_test_probabilities)}

        return all_test_predictions, all_test_probabilities, self.test_labels, results_dict


class OptimizationTCNFeatureSequenceCombinedCopies:
    """Sequential + static data.

    Parameters
    ----------
    model : nn.Module
        TCN model with specified input_dim, hidden_dim, output_dim, batch_size, device.
    loss_fn : torch.nn.modules.loss
        Loss function.
    optimizer : torch.optim
        Optimizer function.
    num_sub_sequences : int
        Number of sub-sequences that make up an entire sequence.
    """

    def __init__(self, model, loss_fn, optimizer, num_sub_sequences, device):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_sub_sequences = num_sub_sequences
        self.device = device
        self.train_predictions = []
        self.val_predictions = []
        self.test_predictions = []
        self.train_probabilities = []
        self.val_probabilities = []
        self.test_probabilities = []
        self.final_train_predictions = []
        self.final_val_predictions = []
        self.final_test_predictions = []
        self.final_train_prob = []
        self.final_val_prob = []
        self.final_test_prob = []

        self.final_max_test_predictions = []
        self.final_max_test_probabilities = []
        self.train_losses = []
        self.val_losses = []
        self.test_labels = []

    def check_if_only_one_boolean(self, *args):
        """Method to check if only one variable in a list of variables has a True value.
        Since Booleans are a subtype of plain integers, you can sum the list of integers
        quite easily and you can also pass true booleans into this function as well.
        """
        return sum(args) == 1

    def get_predictions_binary_model(self, y_pred, train, val, test):
        """Get the predictions and probabilities for the train/val/test batch"""
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, you are " \
                                                     f"now in {len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        # Calculate the probabilities from the raw logits
        output_prob = torch.sigmoid(y_pred)

        pred = output_prob > 0.5  # apply threshold to get class predictions

        # Data is first moved to cpu and then converted to numpy array
        if train:
            self.train_predictions.append(pred.cpu().data.numpy())
            self.train_probabilities.append(output_prob.cpu().data.numpy())
        elif val:
            self.val_predictions.append(pred.cpu().data.numpy())
            self.val_probabilities.append(output_prob.cpu().data.numpy())
        elif test:
            self.test_predictions.append(pred.cpu().data.numpy())
            self.test_probabilities.append(output_prob.cpu().data.numpy())

    def get_mean_prediction_and_probability(self, train, val, test):
        """Get the mean pred and prob over all sub-sequences.

        If a total sequence is 50 time steps and the sub_sequence length is 10 time steps,
        then there are 50 / 10 = 5 sub_sequences. Meaning, that we'll take the mean over the
        5 sub-sequences to obtain the mean pred and prob.

        Returns
        -------
        mean_pred : List[int]
            List of size [batch_size]. Containing the mean prediction for all batch_size rec ids.
        mean_prob : List[float]
            List of size [batch_size]. Containing the mean probability for all batch_size rec ids.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, you " \
                                                     f"are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        if train:
            predictions = self.train_predictions
            probabilities = self.train_probabilities
        elif val:
            predictions = self.val_predictions
            probabilities = self.val_probabilities
        elif test:
            predictions = self.test_predictions
            probabilities = self.test_probabilities

        # Calculate the mean prediction by taking the average over all sub_sequences that make up
        # one total sequence. The variable predictions contains all the sub_sequence predictions
        # belonging to batch_size rec ids.

        mean_pred = list(np.array(np.mean(np.stack(predictions, axis=1), axis=1), dtype=np.int64).flat)
        mean_prob = list(np.array(np.mean(np.stack(probabilities, axis=1), axis=1)).flat)

        max_pred = list(np.array(np.max(np.stack(predictions, axis=1), axis=1), dtype=np.int64).flat)
        max_prob = list(np.array(np.max(np.stack(probabilities, axis=1), axis=1)).flat)

        return mean_pred, mean_prob, max_pred, max_prob

    def obtain_correct_classified_instances(self, t, train, val, test):
        """Obtain the number of correctly classified instances by taking the mean prediction over all
        sub sequences and compare it against the true label.

        Parameters
        ----------
        y : torch.Tensor
            Tensor containing the true labels. Shape [batch_size, output_size].
        t : int
            The time step in the data loader.
        train : Boolean
            If in train mode.
        val : Boolean
            If in validation mode.
        test : Boolean
            If in test mode.
        Returns
        -------
        num_correct : int
            Percentage of change in prediction within the total sequences.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, " \
                                                     f"you are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        if train:
            predictions = self.train_predictions
        elif val:
            predictions = self.val_predictions
        elif test:
            predictions = self.test_predictions

        assert len(predictions) == self.num_sub_sequences, f'The number of predictions on time step {t} ' \
                                                           f'should be {self.num_sub_sequences}, but ' \
                                                           f'is {len(predictions)}'

        mean_pred, mean_prob, max_pred, max_prob = self.get_mean_prediction_and_probability(train, val, test)

        return mean_pred, mean_prob, max_pred, max_prob

    def evaluate_after_entire_sequence(self, y, t, train, val, test):
        """
        After an entire sequence is processed, we want to know the average prediction over the entire
        sequence and use that as a final prediction for the entire sequence.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, " \
                                                     f"you are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"

        # Data is first moved to cpu and then converted to numpy array
        true_label = y.cpu().data.numpy()
        mean_pred, mean_prob, max_pred, max_prob = self.obtain_correct_classified_instances(t,
                                                                                            train=train,
                                                                                            val=val,
                                                                                            test=test)

        # We clear the list with the predictions for the next batch which contains the sequences of new
        # rec ids
        if train:
            self.final_train_predictions.append(mean_pred)
            self.train_predictions.clear()
            self.final_train_prob.append(mean_prob)
            self.train_probabilities.clear()

        elif val:
            self.final_val_predictions.append(mean_pred)
            self.val_predictions.clear()
            self.final_val_prob.append(mean_prob)
            self.val_probabilities.clear()

        elif test:
            self.test_labels.append(true_label)
            self.final_test_predictions.append(mean_pred)
            self.test_predictions.clear()
            self.final_test_prob.append(mean_prob)
            self.test_probabilities.clear()
            self.final_max_test_predictions.append(max_pred)
            self.final_max_test_probabilities.append(max_prob)

        return mean_pred, mean_prob, max_pred, max_prob

    def train_step_combined(self, x_seq, y):
        """Returns loss and prediction."""
        # Sets model to train mode
        self.model.train()

        # Zeroes gradients
        self.optimizer.zero_grad()

        # Forward pass to get output/logits. For BCEwithlogits loss, y_pred will be logits
        y_pred = self.model(x_seq)

        loss = self.loss_fn(y_pred, y)

        if loss.isnan().any():
            print(f'Loss before backward pass: {loss}')

        # Getting gradients w.r.t. parameters
        loss.backward()

        if loss.isnan().any():
            print(f'Loss after backward pass: {loss}')

        # Updates parameters
        self.optimizer.step()

        return loss, y_pred

    def train(self, trial, train_loader_list, test_loader_list: List[DataLoader], params: Dict, n_epochs=50):
        """The model is trained/validated/tested on each sub-sequence up until each entire
        sequence has been processed. The entire sequence length is 28800 time steps and the
        sub_sequence length is 200 time steps. We will make a prediction after each sub-sequence
        has been processed.

        This means that 1 batch contains 200 time steps of batch_size rec ids. As a result, we will
        reset all predictions after 144 batches have been processed. Then we will start processing
        the entire sequences of the next batch_size rec ids and again reset after 144 batches have
        been processed. This process is repeated up until the entire sequence of each rec id has
        been processed.

        As the sequence of each rec id has been truncated/padded to the same length, we will mask
        the padded data points such that it does not contribute to the loss function or predictions.
        For the sequences that have been padded with more than sub_sequence length (=200) time steps,
        this means that the last batches for that rec id contains solely padded values.

        Parameters
        ----------
        trial :
        train_loader_list : List[DataLoader]
        test_loader_list : List[DataLoader]
        params : dict
            Dictionary with the hyperparameters and its values.
        n_epochs : int
            Number of epochs for which you want to train your model.
        """
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(1, n_epochs + 1):
            total_train_loss_epoch = 0.0
            correct_train, total_samples_train = 0, 0
            # Clear at the beginning of every epoch
            self.final_train_prob.clear()
            self.final_val_prob.clear()

            for train_loader in train_loader_list:
                for t, (x_batch, y_batch) in enumerate(train_loader):
                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size_train = x_batch.shape[0]

                    # Reshape it to [batch_size, num_features, sequence length]
                    # !! DO NOT USE .view FOR THIS -> will semantically mess up your data
                    x_batch = x_batch.permute(0, 2, 1)
                    x_batch = x_batch.to(self.device)

                    y_batch = y_batch.to(self.device)

                    total_samples_train += batch_size_train

                    # avg_train_batch_loss contains the value of the average cost of all the training examples of
                    # the current batch
                    avg_train_batch_loss, y_pred = self.train_step_combined(x_batch, y_batch)
                    self.get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                    # After an entire sequence is processed, we clear the predictions and probs
                    if len(self.train_predictions) == self.num_sub_sequences:
                        self.train_predictions.clear()
                        self.train_probabilities.clear()

                    # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                    total_train_loss_epoch += (avg_train_batch_loss.item() * batch_size_train)

            # Calculate average sample train loss for this particular epoch
            avg_sample_train_loss_epoch = total_train_loss_epoch / total_samples_train
            self.train_losses.append(avg_sample_train_loss_epoch)

            self.model.eval()
            torch.autograd.set_detect_anomaly(True)
            with torch.no_grad():
                total_val_loss_epoch = 0.0
                correct_val, total_samples_val = 0, 0
                for val_loader in test_loader_list:
                    for t, (x_val, y_val) in enumerate(val_loader):

                        # The order in the batch MUST be [batch_size, sequence length, num_features]
                        batch_size_val = x_val.shape[0]

                        # Reshape it to [batch_size, num_features, sequence length]
                        # !! DO NOT USE .view FOR THIS -> will semantically mess up your data
                        x_val = x_val.permute(0, 2, 1)
                        x_val = x_val.to(self.device)

                        y_val = y_val.to(self.device)

                        total_samples_val += batch_size_val

                        # forward pass: compute predicted outputs by passing inputs to the model
                        # output shape is [batch_size, output_dim] and contains logits
                        y_pred = self.model(x_val)

                        avg_val_batch_loss = self.loss_fn(y_pred, y_val)

                        self.get_predictions_binary_model(y_pred, train=False, val=True, test=False)

                        if len(self.val_predictions) == self.num_sub_sequences:
                            self.val_predictions.clear()
                            self.val_probabilities.clear()

                        # total_val_loss_epoch accumulates the total loss per validation batch for the entire epoch
                        total_val_loss_epoch += (avg_val_batch_loss.item() * batch_size_val)

                # Calculate average sample validation loss for this particular epoch
                avg_sample_val_loss_epoch = total_val_loss_epoch / total_samples_val
                self.val_losses.append(avg_sample_val_loss_epoch)

            print(f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.4f}\t "
                  f"Validation loss: {avg_sample_val_loss_epoch:.4f}")
            print('-' * 50)

        print('TRAINING COMPLETE')

        return avg_sample_val_loss_epoch, params

    def final_train(self, train_loader_list, feature_name, add_static_data, fold_i, n_epochs=50):
        """The final model will be trained with the optimal hyperparameters on the train+val datset."""
        output_path = file_paths['output_path'] + "/" + "model"

        current_date_and_time = "{:%Y-%m-%d_%H-%M}".format(datetime.datetime.now())

        if add_static_data:
            file_name_output = f'{current_date_and_time}_fold_{fold_i}_best_model_tcn_feature_{feature_name}_combined_copies_seq_final_train'

        if not add_static_data:
            file_name_output = f'{current_date_and_time}_fold_{fold_i}_best_model_tcn_feature_{feature_name}_seq_final_train'

        for epoch in range(1, n_epochs + 1):
            self.final_train_prob.clear()
            total_train_loss_epoch = 0.0
            correct_train, total_samples_train = 0, 0

            for train_loader in train_loader_list:
                for t, (x_batch, y_batch) in enumerate(train_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size = x_batch.shape[0]

                    # Reshape it to [batch_size, num_features, sequence length]
                    # !! DO NOT USE .view FOR THIS -> will semantically mess up your data
                    x_batch = x_batch.permute(0, 2, 1)
                    x_batch = x_batch.to(self.device)

                    total_samples_train += batch_size
                    y_batch = y_batch.to(self.device)

                    # avg_train_batch_loss contains the value of the average cost of all the training examples of
                    # the current batch
                    # y_pred is with the masked predictions
                    avg_train_batch_loss, y_pred = self.train_step_combined(x_batch, y_batch)

                    self.get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                    # After an entire sequence is processed, we want to know the average prediction over the entire
                    # sequence and use that as a final prediction for the entire sequence
                    if len(self.train_predictions) == self.num_sub_sequences:
                        self.train_predictions.clear()
                        self.train_probabilities.clear()

                    # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                    total_train_loss_epoch += (avg_train_batch_loss.item() * batch_size)

            # Calculate average sample train loss for this particular epoch.
            avg_sample_train_loss_epoch = total_train_loss_epoch / total_samples_train
            self.train_losses.append(avg_sample_train_loss_epoch)

            print(f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.10f}")
            print('-' * 50)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn}, f'{output_path}/{file_name_output}.pth')
        print('TRAINING COMPLETE')
        print(f'Model is saved at: {output_path}/{file_name_output}.pth')

    def evaluate(self, test_loader, features_to_use, checkpoint):
        """Evaluate the model by making predictions over each subsequence and after all
        subsequences for each rec id have been processed take the mean over all subsequence predictions."""

        all_test_predictions = []
        all_test_probabilities = []

        model_epoch = checkpoint['epoch']
        print(f"Model was saved at {model_epoch} epochs\n")
        print(f'Loading at epoch {model_epoch} saved model weights...')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        with torch.no_grad():
            correct_test, total_samples_test = 0, 0
            for data_loader in test_loader:
                for t, (x_test, y_test) in enumerate(data_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size_test = x_test.shape[0]

                    # Reshape to [batch_size, num_features, sequence length]
                    x_test = x_test.permute(0, 2, 1)
                    x_test = x_test.to(self.device)
                    y_test = y_test.to(self.device)

                    total_samples_test += batch_size_test

                    # forward pass: compute predicted outputs by passing inputs to the model
                    # output shape is [batch_size, 1] and contains logits
                    y_pred = self.model(x_test)
                    self.get_predictions_binary_model(y_pred, train=False, val=False, test=True)

                    # After the entire sequence has been processed we take the mean prediction of all sub-sequence
                    # predictions
                    if len(self.test_predictions) == self.num_sub_sequences:
                        all_test_predictions.append(self.test_predictions.copy())
                        all_test_probabilities.append(self.test_probabilities.copy())

                        mean_pred, mean_prob, max_pred, max_prob = self.evaluate_after_entire_sequence(y_test, t,
                                                                                                       train=False,
                                                                                                       val=False,
                                                                                                       test=True)

        self.test_labels = list(np.array(np.concatenate([array for array in self.test_labels], axis=0),
                                         dtype=np.int64).flat)
        self.final_test_prob = list(np.array(np.concatenate([array for array in self.final_test_prob], axis=0)).flat)
        self.final_test_predictions = list(np.array(np.concatenate([array for array in self.final_test_predictions],
                                                                   axis=0)).flat)

        self.final_max_test_predictions = list(
            np.array(np.concatenate([array for array in self.final_max_test_predictions], axis=0)).flat)
        self.final_max_test_probabilities = list(
            np.array(np.concatenate([array for array in self.final_max_test_probabilities], axis=0)).flat)

        print(f'Precision score with mean pred: {precision_score(self.test_labels, self.final_test_predictions)}')
        print(f'Recall score with mean pred: {recall_score(self.test_labels, self.final_test_predictions)}')
        print(f'F1 score with mean pred: {f1_score(self.test_labels, self.final_test_predictions)}')
        print(f'AUC score with mean prob: {roc_auc_score(self.test_labels, self.final_test_prob)}')
        print(f'AP score with mean prob: {average_precision_score(self.test_labels, self.final_test_prob)}')

        print(f'Precision score with max pred: {precision_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'Recall score with max pred: {recall_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'F1 score with max pred: {f1_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'AUC score with max prob: {roc_auc_score(self.test_labels, self.final_max_test_probabilities)}')
        print(f'AP score with max prob: {average_precision_score(self.test_labels, self.final_max_test_probabilities)}')

        results_dict = {'precision_mean_pred': precision_score(self.test_labels, self.final_test_predictions),
                        'recall_mean_pred': recall_score(self.test_labels, self.final_test_predictions),
                        'f1_mean_pred': f1_score(self.test_labels, self.final_test_predictions),
                        'auc_mean_prob': roc_auc_score(self.test_labels, self.final_test_prob),
                        'ap_mean_prob': average_precision_score(self.test_labels, self.final_test_prob),
                        'precision_max_pred': precision_score(self.test_labels, self.final_max_test_predictions),
                        'recall_max_pred': recall_score(self.test_labels, self.final_max_test_predictions),
                        'f1_max_pred': f1_score(self.test_labels, self.final_max_test_predictions),
                        'auc_max_prob': roc_auc_score(self.test_labels, self.final_max_test_probabilities),
                        'ap_max_prob': average_precision_score(self.test_labels, self.final_max_test_probabilities)}

        return all_test_predictions, all_test_probabilities, self.test_labels, results_dict


class OptimizationTCNFeatureSequenceCombined:
    """Sequential + static data.

    Parameters
    ----------
    model : nn.Module
        TCN model with specified input_dim, hidden_dim, output_dim, batch_size, device.
    loss_fn : torch.nn.modules.loss
        Loss function.
    optimizer : torch.optim
        Optimizer function.
    num_sub_sequences : int
        Number of sub-sequences that make up an entire sequence.
    """
    def __init__(self, model, loss_fn, optimizer, num_sub_sequences, device):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_sub_sequences = num_sub_sequences
        self.device = device
        self.train_predictions = []
        self.val_predictions = []
        self.test_predictions = []
        self.train_probabilities = []
        self.val_probabilities = []
        self.test_probabilities = []
        self.final_train_predictions = []
        self.final_val_predictions = []
        self.final_test_predictions = []
        self.final_train_prob = []
        self.final_val_prob = []
        self.final_test_prob = []

        self.final_max_val_probabilities = []
        self.final_max_test_predictions = []
        self.final_max_test_probabilities = []
        self.train_losses = []
        self.val_losses = []
        self.val_labels = []
        self.test_labels = []

    def check_if_only_one_boolean(self, *args):
        """Method to check if only one variable in a list of variables has a True value.
        Since Booleans are a subtype of plain integers, you can sum the list of integers
        quite easily and you can also pass true booleans into this function as well.
        """
        return sum(args) == 1

    def get_predictions_binary_model(self, y_pred, train, val, test):
        """Get the predictions and probabilities for the train/val/test batch"""
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, you are " \
                                                     f"now in {len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        # Calculate the probabilities from the raw logits
        output_prob = torch.sigmoid(y_pred)

        pred = output_prob > 0.5  # apply threshold to get class predictions

        # Data is first moved to cpu and then converted to numpy array
        if train:
            self.train_predictions.append(pred.cpu().data.numpy())
            self.train_probabilities.append(output_prob.cpu().data.numpy())
        elif val:
            self.val_predictions.append(pred.cpu().data.numpy())
            self.val_probabilities.append(output_prob.cpu().data.numpy())
        elif test:
            self.test_predictions.append(pred.cpu().data.numpy())
            self.test_probabilities.append(output_prob.cpu().data.numpy())

    def get_mean_prediction_and_probability(self, train, val, test):
        """Get the mean pred and prob over all sub-sequences.

        If a total sequence is 50 time steps and the sub_sequence length is 10 time steps,
        then there are 50 / 10 = 5 sub_sequences. Meaning, that we'll take the mean over the
        5 sub-sequences to obtain the mean pred and prob.

        Returns
        -------
        mean_pred : List[int]
            List of size [batch_size]. Containing the mean prediction for all batch_size rec ids.
        mean_prob : List[float]
            List of size [batch_size]. Containing the mean probability for all batch_size rec ids.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, you " \
                                                     f"are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        if train:
            predictions = self.train_predictions
            probabilities = self.train_probabilities
        elif val:
            predictions = self.val_predictions
            probabilities = self.val_probabilities
        elif test:
            predictions = self.test_predictions
            probabilities = self.test_probabilities

        # Calculate the mean prediction by taking the average over all sub_sequences that make up
        # one total sequence. The variable predictions contains all the sub_sequence predictions
        # belonging to batch_size rec ids.

        mean_pred = list(np.array(np.mean(np.stack(predictions, axis=1), axis=1), dtype=np.int64).flat)
        mean_prob = list(np.array(np.mean(np.stack(probabilities, axis=1), axis=1)).flat)

        max_pred = list(np.array(np.max(np.stack(predictions, axis=1), axis=1), dtype=np.int64).flat)
        max_prob = list(np.array(np.max(np.stack(probabilities, axis=1), axis=1)).flat)

        return mean_pred, mean_prob, max_pred, max_prob

    def obtain_correct_classified_instances(self, t, train, val, test):
        """Obtain the number of correctly classified instances by taking the mean prediction over all
        sub sequences and compare it against the true label.

        Parameters
        ----------
        y : torch.Tensor
            Tensor containing the true labels. Shape [batch_size, output_size].
        t : int
            The time step in the data loader.
        train : Boolean
            If in train mode.
        val : Boolean
            If in validation mode.
        test : Boolean
            If in test mode.
        Returns
        -------
        num_correct : int
            Percentage of change in prediction within the total sequences.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, " \
                                                     f"you are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        if train:
            predictions = self.train_predictions
        elif val:
            predictions = self.val_predictions
        elif test:
            predictions = self.test_predictions

        assert len(predictions) == self.num_sub_sequences, f'The number of predictions on time step {t} ' \
                                                           f'should be {self.num_sub_sequences}, but ' \
                                                           f'is {len(predictions)}'

        mean_pred, mean_prob, max_pred, max_prob = self.get_mean_prediction_and_probability(train, val, test)

        return mean_pred, mean_prob, max_pred, max_prob

    def evaluate_after_entire_sequence(self, y, t, train, val, test):
        """
        After an entire sequence is processed, we want to know the average prediction over the entire
        sequence and use that as a final prediction for the entire sequence.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, " \
                                                     f"you are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"

        # Data is first moved to cpu and then converted to numpy array
        true_label = y.cpu().data.numpy()
        mean_pred, mean_prob, max_pred, max_prob = self.obtain_correct_classified_instances(t,
                                                                                            train=train,
                                                                                            val=val,
                                                                                            test=test)

        # We clear the list with the predictions for the next batch which contains the sequences of new
        # rec ids
        if train:
            self.final_train_predictions.append(mean_pred)
            self.train_predictions.clear()
            self.final_train_prob.append(mean_prob)
            self.train_probabilities.clear()

        elif val:
            self.val_labels.append(true_label)
            self.final_val_predictions.append(mean_pred)
            self.val_predictions.clear()
            self.final_val_prob.append(mean_prob)
            self.val_probabilities.clear()

            self.final_max_val_probabilities.append(max_prob)

        elif test:
            self.test_labels.append(true_label)
            self.final_test_predictions.append(mean_pred)
            self.test_predictions.clear()
            self.final_test_prob.append(mean_prob)
            self.test_probabilities.clear()
            self.final_max_test_predictions.append(max_pred)
            self.final_max_test_probabilities.append(max_prob)

        return mean_pred, mean_prob, max_pred, max_prob

    def train_step_combined(self, x_seq, x_static, y):
        """Returns loss and prediction."""
        # Sets model to train mode
        self.model.train()

        # Zeroes gradients
        self.optimizer.zero_grad()

        # Forward pass to get output/logits. For BCEwithlogits loss, y_pred will be logits
        y_pred = self.model(x_seq, x_static)

        loss = self.loss_fn(y_pred, y)

        if loss.isnan().any():
            print(f'Loss before backward pass: {loss}')

        # Getting gradients w.r.t. parameters
        loss.backward()

        if loss.isnan().any():
            print(f'Loss after backward pass: {loss}')

        # Updates parameters
        self.optimizer.step()

        return loss, y_pred

    def train(self, trial, train_loader_list: List[DataLoader], test_loader_list: List[DataLoader],
              features_to_use, params, n_epochs=50):
        """The model is trained/validated/tested on each sub-sequence up until each entire
        sequence has been processed. The entire sequence length is 28800 time steps and the
        sub_sequence length is 200 time steps. We will make a prediction after each sub-sequence
        has been processed.

        This means that 1 batch contains 200 time steps of batch_size rec ids. As a result, we will
        reset all predictions after 144 batches have been processed. Then we will start processing
        the entire sequences of the next batch_size rec ids and again reset after 144 batches have
        been processed. This process is repeated up until the entire sequence of each rec id has
        been processed.

        As the sequence of each rec id has been truncated/padded to the same length, we will mask
        the padded data points such that it does not contribute to the loss function or predictions.
        For the sequences that have been padded with more than sub_sequence length (=200) time steps,
        this means that the last batches for that rec id contains solely padded values.

        Parameters
        ----------
        trial :
        df_signals : pd.DataFrame
            Original signal dataframe with the features_to_use present.
        X_train : pd.DataFrame
            Dataframe that contains the train samples.
        X_val : pd.DataFrame
            Dataframe that contains the validation samples.
        features_to_use : List[str]
            List with the names of features you want to use for modeling.
        params : dict
            Dictionary with the hyperparameters and its values.
        n_epochs : int
            Number of epochs for which you want to train your model.
        """
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(1, n_epochs + 1):
            total_train_loss_epoch = 0.0
            correct_train, total_samples_train = 0, 0
            # Clear at the beginning of every epoch
            self.final_train_prob.clear()
            self.final_val_prob.clear()
            self.val_labels.clear()

            self.train_losses.clear()
            self.val_losses.clear()

            self.final_val_predictions.clear()
            self.final_max_val_probabilities.clear()

            for train_loader in train_loader_list:
                for t, (x_batch, y_batch) in enumerate(train_loader):
                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size_train = x_batch.shape[0]

                    # Reshape it to [batch_size, num_features, sequence length]
                    # !! DO NOT USE .view FOR THIS -> will semantically mess up your data
                    x_batch = x_batch.permute(0, 2, 1)

                    # Split the sequential and static data.
                    # The seq data are the first len(features_to_use) columns and the static
                    # data is the other part
                    x_batch_seq = x_batch[:, 0:len(features_to_use), :].to(self.device)

                    x_batch_static = x_batch[:, len(features_to_use):, :]
                    # The static data is now the same length as the number of time steps
                    # in the seq data. Meaning, if there are 10 time steps in the seq data,
                    # then the static data is also copied for 10 time steps.
                    # As we don't want copies of the static data along all time steps,
                    # but just 1 time step we keep the first value of the static data
                    # for each sample in the batch.
                    # This is a bit of a quick fix for now, it would be neater to not
                    # have the copies in the first place
                    x_batch_static = x_batch_static[:, :, 0].to(self.device)
                    y_batch = y_batch.to(self.device)

                    total_samples_train += batch_size_train

                    # avg_train_batch_loss contains the value of the average cost of all the training examples of
                    # the current batch
                    avg_train_batch_loss, y_pred = self.train_step_combined(x_batch_seq, x_batch_static, y_batch)
                    self.get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                    # After an entire sequence is processed, we clear the predictions and probs
                    if len(self.train_predictions) == self.num_sub_sequences:
                        self.train_predictions.clear()
                        self.train_probabilities.clear()

                    # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                    total_train_loss_epoch += (avg_train_batch_loss.item() * batch_size_train)

            # Calculate average sample train loss for this particular epoch
            avg_sample_train_loss_epoch = total_train_loss_epoch / total_samples_train
            self.train_losses.append(avg_sample_train_loss_epoch)

            self.model.eval()
            torch.autograd.set_detect_anomaly(True)
            with torch.no_grad():
                total_val_loss_epoch = 0.0
                correct_val, total_samples_val = 0, 0
                all_val_probabilities = []

                for val_loader in test_loader_list:
                    for t, (x_val, y_val) in enumerate(val_loader):

                        # The order in the batch MUST be [batch_size, sequence length, num_features]
                        batch_size_val = x_val.shape[0]

                        # Reshape it to [batch_size, num_features, sequence length]
                        # !! DO NOT USE .view FOR THIS -> will semantically mess up your data
                        x_val = x_val.permute(0, 2, 1)

                        # Split the sequential and static data.
                        # The seq data are the first len(features_to_use) columns and the static
                        # data is the other part
                        x_val_seq = x_val[:, 0:len(features_to_use), :].to(self.device)

                        x_val_static = x_val[:, len(features_to_use):, :]

                        x_val_static = x_val_static[:, :, 0].to(self.device)
                        y_val = y_val.to(self.device)

                        total_samples_val += batch_size_val

                        # forward pass: compute predicted outputs by passing inputs to the model
                        # output shape is [batch_size, output_dim] and contains logits
                        y_pred = self.model(x_val_seq, x_val_static)

                        avg_val_batch_loss = self.loss_fn(y_pred, y_val)

                        self.get_predictions_binary_model(y_pred, train=False, val=True, test=False)

                        if len(self.val_predictions) == self.num_sub_sequences:
                            all_val_probabilities.append(self.val_probabilities.copy())
                            mean_pred, mean_prob, max_pred, max_prob = self.evaluate_after_entire_sequence(y_val, t,
                                                                                                           train=False,
                                                                                                           val=True,
                                                                                                           test=False)
                            # self.val_predictions.clear()
                            # self.val_probabilities.clear()

                        # total_val_loss_epoch accumulates the total loss per validation batch for the entire epoch
                        total_val_loss_epoch += (avg_val_batch_loss.item() * batch_size_val)

                # Calculate average sample validation loss for this particular epoch
                avg_sample_val_loss_epoch = total_val_loss_epoch / total_samples_val
                self.val_losses.append(avg_sample_val_loss_epoch)
                val_labels_final = list(np.array(np.concatenate([array for array in self.val_labels], axis=0),
                                                dtype=np.int64).flat)

                final_val_prob_final = list(np.array(np.concatenate([array for array in self.final_val_prob],
                                                                   axis=0)).flat)

                final_max_val_probabilities_final = list(
                    np.array(np.concatenate([array for array in self.final_max_val_probabilities], axis=0)).flat)

                auc_mean_val = roc_auc_score(val_labels_final, final_val_prob_final)
                auc_max_val = roc_auc_score(val_labels_final, final_max_val_probabilities_final)

            print(f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.4f}\t "
                  f"Validation loss: {avg_sample_val_loss_epoch:.4f}\t AUC mean val: {auc_mean_val} \t AUC max val: {auc_max_val}")
            print('-' * 50)

        print('TRAINING COMPLETE')

        return auc_mean_val, auc_max_val, params

    def final_train(self, train_loader_list, features_to_use, feature_name, add_static_data, fold_i, n_epochs=50):
        """The final model will be trained with the optimal hyperparameters after nested cross validation."""
        output_path = file_paths['output_path'] + "/" + "model"

        current_date_and_time = "{:%Y-%m-%d_%H-%M}".format(datetime.datetime.now())

        if add_static_data:
            file_name_output = f'{current_date_and_time}_fold_{fold_i}_best_model_tcn_feature_{feature_name}_combined_seq_final_train'

        if not add_static_data:
            file_name_output = f'{current_date_and_time}_fold_{fold_i}_best_model_tcn_feature_{feature_name}_seq_final_train'

        for epoch in range(1, n_epochs + 1):
            self.final_train_prob.clear()
            total_train_loss_epoch = 0.0
            correct_train, total_samples_train = 0, 0

            for train_loader in train_loader_list:
                for t, (x_batch, y_batch) in enumerate(train_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size = x_batch.shape[0]

                    # Reshape it to [batch_size, num_features, sequence length]
                    # !! DO NOT USE .view FOR THIS -> will semantically mess up your data
                    x_batch = x_batch.permute(0, 2, 1)

                    total_samples_train += batch_size

                    # Split the sequential and static data.
                    # The seq data are the first len(features_to_use) columns and the static
                    # data is the other part
                    x_batch_seq = x_batch[:, 0:len(features_to_use), :].to(self.device)

                    x_batch_static = x_batch[:, len(features_to_use):, :]

                    x_batch_static = x_batch_static[:, :, 0].to(self.device)
                    y_batch = y_batch.to(self.device)

                    # avg_train_batch_loss contains the value of the average cost of all the training examples of
                    # the current batch
                    # y_pred is with the masked predictions
                    avg_train_batch_loss, y_pred = self.train_step_combined(x_batch_seq, x_batch_static, y_batch)

                    self.get_predictions_binary_model(y_pred, train=True, val=False, test=False)
                    print(f'Train probs: {self.train_probabilities}')
                    print(f'Y batch: {y_batch}')

                    # After an entire sequence is processed, we want to know the average prediction over the entire
                    # sequence and use that as a final prediction for the entire sequence
                    if len(self.train_predictions) == self.num_sub_sequences:
                        self.train_predictions.clear()
                        self.train_probabilities.clear()

                    # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                    total_train_loss_epoch += (avg_train_batch_loss.item() * batch_size)

            # Calculate average sample train loss for this particular epoch.
            avg_sample_train_loss_epoch = total_train_loss_epoch / total_samples_train
            self.train_losses.append(avg_sample_train_loss_epoch)

            print(f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.10f}")
            print('-' * 50)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn}, f'{output_path}/{file_name_output}.pth')
        print('TRAINING COMPLETE')
        print(f'Model is saved at: {output_path}/{file_name_output}.pth')

    def evaluate(self, test_loader, features_to_use, checkpoint):
        """Evaluate the model by making predictions over each subsequence and after all
        subsequences for each rec id have been processed take the mean over all subsequence predictions."""

        all_test_predictions = []
        all_test_probabilities = []

        model_epoch = checkpoint['epoch']
        print(f"Model was saved at {model_epoch} epochs\n")
        print(f'Loading at epoch {model_epoch} saved model weights...')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        with torch.no_grad():
            correct_test, total_samples_test = 0, 0
            for data_loader in test_loader:
                for t, (x_test, y_test) in enumerate(data_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size_test = x_test.shape[0]

                    # Reshape to [batch_size, num_features, sequence length]
                    x_test = x_test.permute(0, 2, 1)

                    # Split the sequential and static data.
                    # The seq data are the first len(features_to_use) columns and the static
                    # data is the other part
                    x_test_seq = x_test[:, 0:len(features_to_use), :].to(self.device)

                    x_test_static = x_test[:, len(features_to_use):, :]
                    # The static data is now the same length as the number of time steps
                    # in the seq data. Meaning, if there are 10 time steps in the seq data,
                    # then the static data is also copied for 10 time steps.
                    # As we don't want copies of the static data along all time steps,
                    # but just 1 time step we keep the first value of the static data
                    # for each sample in the batch.
                    # This is a bit of a quick fix for now, it would be neater to not
                    # have the copies in the first place
                    x_test_static = x_test_static[:, :, 0].to(self.device)
                    y_test = y_test.to(self.device)

                    total_samples_test += batch_size_test

                    # forward pass: compute predicted outputs by passing inputs to the model
                    # output shape is [batch_size, 1] and contains logits
                    y_pred = self.model(x_test_seq, x_test_static)
                    self.get_predictions_binary_model(y_pred, train=False, val=False, test=True)

                    # After the entire sequence has been processed we take the mean prediction of all sub-sequence
                    # predictions
                    if len(self.test_predictions) == self.num_sub_sequences:
                        all_test_predictions.append(self.test_predictions.copy())
                        all_test_probabilities.append(self.test_probabilities.copy())

                        mean_pred, mean_prob, max_pred, max_prob = self.evaluate_after_entire_sequence(y_test, t,
                                                                                                       train=False,
                                                                                                       val=False,
                                                                                                       test=True)

        self.test_labels = list(np.array(np.concatenate([array for array in self.test_labels], axis=0),
                                         dtype=np.int64).flat)
        self.final_test_prob = list(np.array(np.concatenate([array for array in self.final_test_prob], axis=0)).flat)
        self.final_test_predictions = list(np.array(np.concatenate([array for array in self.final_test_predictions],
                                                                   axis=0)).flat)

        self.final_max_test_predictions = list(
            np.array(np.concatenate([array for array in self.final_max_test_predictions], axis=0)).flat)
        self.final_max_test_probabilities = list(
            np.array(np.concatenate([array for array in self.final_max_test_probabilities], axis=0)).flat)

        print(f'Precision score with mean pred: {precision_score(self.test_labels, self.final_test_predictions)}')
        print(f'Recall score with mean pred: {recall_score(self.test_labels, self.final_test_predictions)}')
        print(f'F1 score with mean pred: {f1_score(self.test_labels, self.final_test_predictions)}')
        print(f'AUC score with mean prob: {roc_auc_score(self.test_labels, self.final_test_prob)}')
        print(f'AP score with mean prob: {average_precision_score(self.test_labels, self.final_test_prob)}')

        print(f'Precision score with max pred: {precision_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'Recall score with max pred: {recall_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'F1 score with max pred: {f1_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'AUC score with max prob: {roc_auc_score(self.test_labels, self.final_max_test_probabilities)}')
        print(f'AP score with max prob: {average_precision_score(self.test_labels, self.final_max_test_probabilities)}')

        results_dict = {'precision_mean_pred': precision_score(self.test_labels, self.final_test_predictions),
                        'recall_mean_pred': recall_score(self.test_labels, self.final_test_predictions),
                        'f1_mean_pred': f1_score(self.test_labels, self.final_test_predictions),
                        'auc_mean_prob': roc_auc_score(self.test_labels, self.final_test_prob),
                        'ap_mean_prob': average_precision_score(self.test_labels, self.final_test_prob),
                        'precision_max_pred': precision_score(self.test_labels, self.final_max_test_predictions),
                        'recall_max_pred': recall_score(self.test_labels, self.final_max_test_predictions),
                        'f1_max_pred': f1_score(self.test_labels, self.final_max_test_predictions),
                        'auc_max_prob': roc_auc_score(self.test_labels, self.final_max_test_probabilities),
                        'ap_max_prob': average_precision_score(self.test_labels, self.final_max_test_probabilities)}

        return all_test_predictions, all_test_probabilities, self.test_labels, results_dict


class OptimizationTCNFeatureSequence:
    """Train, optimize and evaluate a TCN model for a long input sequence where the input
    sequences is split into multiple sub-sequences and a prediction is made over each sub-sequence.

    The number of sub-sequences needed to process an entire sequence is denoted by the parameter
    num_sub_sequences. Each sequence has been padded or truncated to the same length. We will mask
    the padded data points such that it does not contribute to the loss function.

    The final prediction will be made by making a prediction for each sub-sequence and at the end take
    the mean over all sub-sequence predictions.

    Parameters
    ----------
    model : nn.Module
        TCN model with specified input_dim, hidden_dim, output_dim, batch_size, device,
        layer_dim, batch_first.
    loss_fn : torch.nn.modules.loss
        Loss function.
    optimizer : torch.optim
        Optimizer function.
    num_sub_sequences : int
        Number of sub-sequences that make up an entire sequence.
    """

    def __init__(self, model, loss_fn, optimizer, num_sub_sequences, device):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_sub_sequences = num_sub_sequences
        self.device = device
        self.train_predictions = []
        self.val_predictions = []
        self.test_predictions = []
        self.train_probabilities = []
        self.val_probabilities = []
        self.test_probabilities = []
        self.final_train_predictions = []
        self.final_val_predictions = []
        self.final_test_predictions = []
        self.final_train_prob = []
        self.final_val_prob = []
        self.final_test_prob = []

        self.final_max_val_probabilities = []
        self.final_max_test_predictions = []
        self.final_max_test_probabilities = []
        self.train_losses = []
        self.val_losses = []
        self.val_labels = []
        self.test_labels = []

    def check_if_only_one_boolean(self, *args):
        """Method to check if only one variable in a list of variables has a True value.
        Since Booleans are a subtype of plain integers, you can sum the list of integers
        quite easily and you can also pass true booleans into this function as well.
        """
        return sum(args) == 1

    def get_predictions_binary_model(self, y_pred, train, val, test):
        """Get the predictions and probabilities for the train/val/test batch"""
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, you are " \
                                                     f"now in {len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        # Calculate the probabilities from the raw logits
        output_prob = torch.sigmoid(y_pred)

        pred = output_prob > 0.5  # apply threshold to get class predictions

        # Data is first moved to cpu and then converted to numpy array
        if train:
            self.train_predictions.append(pred.cpu().data.numpy())
            self.train_probabilities.append(output_prob.cpu().data.numpy())
        elif val:
            self.val_predictions.append(pred.cpu().data.numpy())
            self.val_probabilities.append(output_prob.cpu().data.numpy())
        elif test:
            self.test_predictions.append(pred.cpu().data.numpy())
            self.test_probabilities.append(output_prob.cpu().data.numpy())

    def get_mean_prediction_and_probability(self, train, val, test):
        """Get the mean pred and prob over all sub-sequences.

        If a total sequence is 50 time steps and the sub_sequence length is 10 time steps,
        then there are 50 / 10 = 5 sub_sequences. Meaning, that we'll take the mean over the
        5 sub-sequences to obtain the mean pred and prob.

        Returns
        -------
        mean_pred : List[int]
            List of size [batch_size]. Containing the mean prediction for all batch_size rec ids.
        mean_prob : List[float]
            List of size [batch_size]. Containing the mean probability for all batch_size rec ids.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, you " \
                                                     f"are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        if train:
            predictions = self.train_predictions
            probabilities = self.train_probabilities
        elif val:
            predictions = self.val_predictions
            probabilities = self.val_probabilities
        elif test:
            predictions = self.test_predictions
            probabilities = self.test_probabilities

        # Calculate the mean prediction by taking the average over all sub_sequences that make up
        # one total sequence. The variable predictions contains all the sub_sequence predictions
        # belonging to batch_size rec ids.

        mean_pred = list(np.array(np.mean(np.stack(predictions, axis=1), axis=1), dtype=np.int64).flat)
        mean_prob = list(np.array(np.mean(np.stack(probabilities, axis=1), axis=1)).flat)

        max_pred = list(np.array(np.max(np.stack(predictions, axis=1), axis=1), dtype=np.int64).flat)
        max_prob = list(np.array(np.max(np.stack(probabilities, axis=1), axis=1)).flat)

        return mean_pred, mean_prob, max_pred, max_prob

    def obtain_correct_classified_instances(self, t, train, val, test):
        """Obtain the number of correctly classified instances by taking the mean prediction over all
        sub sequences and compare it against the true label.

        Parameters
        ----------
        y : torch.Tensor
            Tensor containing the true labels. Shape [batch_size, output_size].
        t : int
            The time step in the data loader.
        train : Boolean
            If in train mode.
        val : Boolean
            If in validation mode.
        test : Boolean
            If in test mode.
        Returns
        -------
        num_correct : int
            Percentage of change in prediction within the total sequences.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, " \
                                                     f"you are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"
        if train:
            predictions = self.train_predictions
        elif val:
            predictions = self.val_predictions
        elif test:
            predictions = self.test_predictions

        assert len(predictions) == self.num_sub_sequences, f'The number of predictions on time step {t} ' \
                                                           f'should be {self.num_sub_sequences}, but ' \
                                                           f'is {len(predictions)}'

        mean_pred, mean_prob, max_pred, max_prob = self.get_mean_prediction_and_probability(train, val, test)

        return mean_pred, mean_prob, max_pred, max_prob

    def evaluate_after_entire_sequence(self, y, t, train, val, test):
        """
        After an entire sequence is processed, we want to know the average prediction over the entire
        sequence and use that as a final prediction for the entire sequence.
        """
        assert self.check_if_only_one_boolean(train, val,
                                              test), f"You can only be in either one of train/val/test mode, " \
                                                     f"you are now in " \
                                                     f"{len([var for var in [train, val, test] if var == True])} " \
                                                     f"modes"

        # Data is first moved to cpu and then converted to numpy array
        true_label = y.cpu().data.numpy()
        mean_pred, mean_prob, max_pred, max_prob = self.obtain_correct_classified_instances(t,
                                                                                            train=train,
                                                                                            val=val,
                                                                                            test=test)

        # We clear the list with the predictions for the next batch which contains the sequences of new
        # rec ids
        if train:
            self.final_train_predictions.append(mean_pred)
            self.train_predictions.clear()
            self.final_train_prob.append(mean_prob)
            self.train_probabilities.clear()

        elif val:
            self.val_labels.append(true_label)
            self.final_val_predictions.append(mean_pred)
            self.val_predictions.clear()
            self.final_val_prob.append(mean_prob)
            self.val_probabilities.clear()

            self.final_max_val_probabilities.append(max_prob)

        elif test:
            self.test_labels.append(true_label)
            self.final_test_predictions.append(mean_pred)
            self.test_predictions.clear()
            self.final_test_prob.append(mean_prob)
            self.test_probabilities.clear()
            self.final_max_test_predictions.append(max_pred)
            self.final_max_test_probabilities.append(max_prob)

        return mean_pred, mean_prob, max_pred, max_prob

    def train_step(self, x, y):
        """Returns loss and prediction."""
        # Sets model to train mode
        self.model.train()

        # Zeroes gradients
        self.optimizer.zero_grad()

        # Forward pass to get output/logits. For BCEwithlogits loss, y_pred will be logits
        y_pred = self.model(x)

        loss = self.loss_fn(y_pred, y)

        if loss.isnan().any():
            print(f'Loss before backward pass: {loss}')

        # Getting gradients w.r.t. parameters
        loss.backward()

        if loss.isnan().any():
            print(f'Loss after backward pass: {loss}')

        # Updates parameters
        self.optimizer.step()

        return loss, y_pred

    def train(self, trial, train_loader_list: List[DataLoader], test_loader_list: List[DataLoader],
              params, n_epochs=50):
        """The model is trained/validated/tested on each sub-sequence up until each entire
        sequence has been processed. The entire sequence length is 28800 time steps and the
        sub_sequence length is 200 time steps. We will make a prediction after each sub-sequence
        has been processed.

        This means that 1 batch contains 200 time steps of batch_size rec ids. As a result, we will
        reset all predictions after 144 batches have been processed. Then we will start processing
        the entire sequences of the next batch_size rec ids and again reset after 144 batches have
        been processed. This process is repeated up until the entire sequence of each rec id has
        been processed.

        As the sequence of each rec id has been truncated/padded to the same length, we will mask
        the padded data points such that it does not contribute to the loss function or predictions.
        For the sequences that have been padded with more than sub_sequence length (=200) time steps,
        this means that the last batches for that rec id contains solely padded values.

        Parameters
        ----------
        trial :
        train_loader_list : List[DataLoader]
        test_loader_list : List[DataLoader]
        params : dict
            Dictionary with the hyperparameters and its values.
        n_epochs : int
            Number of epochs for which you want to train your model.
        """
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(1, n_epochs + 1):
            total_train_loss_epoch = 0.0
            correct_train, total_samples_train = 0, 0
            # Clear at the beginning of every epoch
            self.final_train_prob.clear()
            self.final_val_prob.clear()
            self.val_labels.clear()

            self.train_losses.clear()
            self.val_losses.clear()

            self.final_val_predictions.clear()
            self.final_max_val_probabilities.clear()

            for train_loader in train_loader_list:
                for t, (x_batch, y_batch) in enumerate(train_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size_train = x_batch.shape[0]

                    # Reshape it to [batch_size, num_features, sequence length]
                    # !! DO NOT USE .view FOR THIS -> will semantically mess up your data
                    x_batch = x_batch.permute(0, 2, 1)

                    total_samples_train += batch_size_train

                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # avg_train_batch_loss contains the value of the average cost of all the training examples of
                    # the current batch
                    avg_train_batch_loss, y_pred = self.train_step(x_batch, y_batch)
                    self.get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                    # After an entire sequence is processed, we clear the predictions and probs
                    if len(self.train_predictions) == self.num_sub_sequences:
                        self.train_predictions.clear()
                        self.train_probabilities.clear()

                    # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                    total_train_loss_epoch += (avg_train_batch_loss.item() * batch_size_train)

            # Calculate average sample train loss for this particular epoch
            avg_sample_train_loss_epoch = total_train_loss_epoch / total_samples_train
            self.train_losses.append(avg_sample_train_loss_epoch)

            self.model.eval()
            torch.autograd.set_detect_anomaly(True)
            with torch.no_grad():
                total_val_loss_epoch = 0.0
                correct_val, total_samples_val = 0, 0
                all_val_probabilities = []

                for val_loader in test_loader_list:
                    for t, (x_val, y_val) in enumerate(val_loader):

                        # The order in the batch MUST be [batch_size, sequence length, num_features]
                        batch_size_val = x_val.shape[0]

                        # Reshape it to [batch_size, num_features, sequence length]
                        # !! DO NOT USE .view FOR THIS -> will semantically mess up your data
                        x_val = x_val.permute(0, 2, 1)
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)

                        total_samples_val += batch_size_val

                        # forward pass: compute predicted outputs by passing inputs to the model
                        # output shape is [batch_size, output_dim] and contains logits
                        y_pred = self.model(x_val)

                        avg_val_batch_loss = self.loss_fn(y_pred, y_val)

                        self.get_predictions_binary_model(y_pred, train=False, val=True, test=False)

                        if len(self.val_predictions) == self.num_sub_sequences:
                            all_val_probabilities.append(self.val_probabilities.copy())
                            mean_pred, mean_prob, max_pred, max_prob = self.evaluate_after_entire_sequence(y_val, t,
                                                                                                           train=False,
                                                                                                           val=True,
                                                                                                           test=False)
                            # self.val_predictions.clear()
                            # self.val_probabilities.clear()

                        # total_val_loss_epoch accumulates the total loss per validation batch for the entire epoch
                        total_val_loss_epoch += (avg_val_batch_loss.item() * batch_size_val)

                # Calculate average sample validation loss for this particular epoch
                avg_sample_val_loss_epoch = total_val_loss_epoch / total_samples_val
                self.val_losses.append(avg_sample_val_loss_epoch)
                val_labels_final = list(np.array(np.concatenate([array for array in self.val_labels], axis=0),
                                                dtype=np.int64).flat)

                final_val_prob_final = list(np.array(np.concatenate([array for array in self.final_val_prob],
                                                                   axis=0)).flat)

                final_max_val_probabilities_final = list(
                    np.array(np.concatenate([array for array in self.final_max_val_probabilities], axis=0)).flat)

                auc_mean_val = roc_auc_score(val_labels_final, final_val_prob_final)
                auc_max_val = roc_auc_score(val_labels_final, final_max_val_probabilities_final)

            print(f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.4f}\t "
                  f"Validation loss: {avg_sample_val_loss_epoch:.4f}\t AUC mean val: {auc_mean_val}\t AUC max val: {auc_max_val}")
            print('-' * 50)

        print('TRAINING COMPLETE')

        return auc_mean_val, auc_max_val, params

    def final_train(self, train_loader_list, feature_name, add_static_data, fold_i, n_epochs=50):
        """The final model will be trained with the optimal hyperparameters on the train+val datset."""
        output_path = file_paths['output_path'] + "/" + "model"
        current_date_and_time = "{:%Y-%m-%d_%H-%M}".format(datetime.datetime.now())

        if add_static_data:
            file_name_output = f'{current_date_and_time}_fold_{fold_i}_best_model_tcn_feature_{feature_name}_combined_seq_final_train'

        if not add_static_data:
            file_name_output = f'{current_date_and_time}_fold_{fold_i}_best_model_tcn_feature_{feature_name}_seq_final_train'

        for epoch in range(1, n_epochs + 1):
            self.final_train_prob.clear()
            total_train_loss_epoch = 0.0
            correct_train, total_samples_train = 0, 0

            for train_loader in train_loader_list:
                for t, (x_batch, y_batch) in enumerate(train_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size = x_batch.shape[0]

                    # Reshape it to [batch_size, num_features, sequence length]
                    # !! DO NOT USE .view FOR THIS -> will semantically mess up your data
                    x_batch = x_batch.permute(0, 2, 1)

                    total_samples_train += batch_size

                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # avg_train_batch_loss contains the value of the average cost of all the training examples of
                    # the current batch
                    # y_pred is with the masked predictions
                    avg_train_batch_loss, y_pred = self.train_step(x_batch, y_batch)

                    self.get_predictions_binary_model(y_pred, train=True, val=False, test=False)

                    # After an entire sequence is processed, we want to know the average prediction over the entire
                    # sequence and use that as a final prediction for the entire sequence
                    if len(self.train_predictions) == self.num_sub_sequences:
                        self.train_predictions.clear()
                        self.train_probabilities.clear()

                    # total_train_loss_epoch accumulates the total loss per train batch for each entire epoch
                    total_train_loss_epoch += (avg_train_batch_loss.item() * batch_size)

            # Calculate average sample train loss for this particular epoch.
            avg_sample_train_loss_epoch = total_train_loss_epoch / total_samples_train
            self.train_losses.append(avg_sample_train_loss_epoch)

            print(f"Epoch [{epoch}/{n_epochs}] Training loss: {avg_sample_train_loss_epoch:.10f}")
            print('-' * 50)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn}, f'{output_path}/{file_name_output}.pth')
        print('TRAINING COMPLETE')
        print(f'Model is saved at: {output_path}/{file_name_output}.pth')

    def evaluate(self, test_loader, checkpoint):
        """Evaluate the model by making predictions over each subsequence and after all
        subsequences for each rec id have been processed take the mean over all subsequence predictions."""

        all_test_predictions = []
        all_test_probabilities = []

        model_epoch = checkpoint['epoch']
        print(f"Model was saved at {model_epoch} epochs\n")
        print(f'Loading at epoch {model_epoch} saved model weights...')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        with torch.no_grad():
            correct_test, total_samples_test = 0, 0
            for data_loader in test_loader:
                for t, (x_test, y_test) in enumerate(data_loader):

                    # The order in the batch MUST be [batch_size, sequence length, num_features]
                    batch_size_test = x_test.shape[0]

                    # Reshape to [batch_size, num_features, sequence length]
                    x_test = x_test.permute(0, 2, 1)
                    x_test = x_test.to(self.device)
                    y_test = y_test.to(self.device)

                    total_samples_test += batch_size_test

                    # forward pass: compute predicted outputs by passing inputs to the model
                    # output shape is [batch_size, 1] and contains logits
                    y_pred = self.model(x_test)

                    self.get_predictions_binary_model(y_pred, train=False, val=False, test=True)

                    # After the entire sequence has been processed we take the mean prediction of all sub-sequence
                    # predictions
                    if len(self.test_predictions) == self.num_sub_sequences:
                        all_test_predictions.append(self.test_predictions.copy())
                        all_test_probabilities.append(self.test_probabilities.copy())

                        mean_pred, mean_prob, max_pred, max_prob = self.evaluate_after_entire_sequence(y_test, t,
                                                                                                       train=False,
                                                                                                       val=False,
                                                                                                       test=True)

        self.test_labels = list(np.array(np.concatenate([array for array in self.test_labels], axis=0),
                                         dtype=np.int64).flat)
        self.final_test_prob = list(np.array(np.concatenate([array for array in self.final_test_prob], axis=0)).flat)
        self.final_test_predictions = list(np.array(np.concatenate([array for array in self.final_test_predictions],
                                                                   axis=0)).flat)

        self.final_max_test_predictions = list(
            np.array(np.concatenate([array for array in self.final_max_test_predictions], axis=0)).flat)
        self.final_max_test_probabilities = list(
            np.array(np.concatenate([array for array in self.final_max_test_probabilities], axis=0)).flat)

        print(f'Precision score with mean pred: {precision_score(self.test_labels, self.final_test_predictions)}')
        print(f'Recall score with mean pred: {recall_score(self.test_labels, self.final_test_predictions)}')
        print(f'Specificity score with mean pred: {recall_score(np.logical_not(self.test_labels), np.logical_not(self.final_test_predictions))}')
        print(f'F1 score with mean pred: {f1_score(self.test_labels, self.final_test_predictions)}')
        print(f'AUC score with mean prob: {roc_auc_score(self.test_labels, self.final_test_prob)}')
        print(f'AP score with mean prob: {average_precision_score(self.test_labels, self.final_test_prob)}')

        print(f'Precision score with max pred: {precision_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'Recall score with max pred: {recall_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'Specificity score with max pred: {recall_score(np.logical_not(self.test_labels), np.logical_not(self.final_max_test_predictions))}')
        print(f'F1 score with max pred: {f1_score(self.test_labels, self.final_max_test_predictions)}')
        print(f'AUC score with max prob: {roc_auc_score(self.test_labels, self.final_max_test_probabilities)}')
        print(f'AP score with max prob: {average_precision_score(self.test_labels, self.final_max_test_probabilities)}')

        results_dict = {f'precision_mean_pred': precision_score(self.test_labels, self.final_test_predictions),
                        f'recall_mean_pred': recall_score(self.test_labels, self.final_test_predictions),
                        f'specificity_mean_pred': recall_score(np.logical_not(self.test_labels),
                                                               np.logical_not(self.final_test_predictions)),
                        f'f1_mean_pred': f1_score(self.test_labels, self.final_test_predictions),
                        f'auc_mean_prob': roc_auc_score(self.test_labels, self.final_test_prob),
                        f'ap_mean_prob': average_precision_score(self.test_labels, self.final_test_prob),
                        f'precision_max_pred': precision_score(self.test_labels, self.final_max_test_predictions),
                        f'recall_max_pred': recall_score(self.test_labels, self.final_max_test_predictions),
                        f'specificity_max_pred': recall_score(np.logical_not(self.test_labels),
                                                              np.logical_not(self.final_max_test_predictions)),
                        f'f1_max_pred': f1_score(self.test_labels, self.final_max_test_predictions),
                        f'auc_max_prob': roc_auc_score(self.test_labels, self.final_max_test_probabilities),
                        f'ap_max_prob': average_precision_score(self.test_labels, self.final_max_test_probabilities)}

        return all_test_predictions, all_test_probabilities, self.test_labels, results_dict


class ObjectiveLSTMFeatureCombinedModel(object):
    def __init__(self, trial, x_train, x_test, x_train_processed, x_test_processed, y_train_processed, y_test_processed,
                 rec_ids_train_inner, rec_ids_test_inner, rec_ids_train_outer, rec_ids_test_outer, pos_weight,
                 feature_name, num_sub_sequences, features_to_use, features_to_use_static, out_file,
                 outer_fold, inner_fold):
        self.x_train = x_train
        self.x_test = x_test
        self.x_train_processed = x_train_processed
        self.x_test_processed = x_test_processed
        self.y_train_processed = y_train_processed
        self.y_test_processed = y_test_processed
        self.rec_ids_train_inner = rec_ids_train_inner
        self.rec_ids_test_inner = rec_ids_test_inner
        self.rec_ids_train_outer = rec_ids_train_outer
        self.rec_ids_test_outer = rec_ids_test_outer
        self.pos_weight = pos_weight
        self.feature_name = feature_name
        self.num_sub_sequences = num_sub_sequences
        self.features_to_use = features_to_use
        self.features_to_use_static = features_to_use_static
        self.out_file = out_file
        self.num_static_features = len(self.features_to_use_static)
        self.outer_fold = outer_fold
        self.inner_fold = inner_fold

        self.auc = self.calculate_auc(trial)

    # Build a model by implementing define-by-run design from Optuna
    def build_model_custom(self, trial, hidden_dim, params):
        # These are the possible optional models we can construct here:

        # The last part of the model before building this optional part was a linear layer and
        # these are the options to go from there:

        # 1. Linear layer -> BatchNorm -> activation layer -> Linear Layer -> activation layer
        # 2. Linear layer -> BatchNorm -> activation layer
        # 3. Linear layer -> activation layer -> BatchNorm -> Linear Layer -> activation layer
        # 4. Linear layer -> activation layer -> BatchNorm
        # 5. Linear layer -> activation layer -> dropout -> Linear Layer -> activation layer
        # 6. Linear layer -> activation layer -> dropout
        # 7. Linear layer -> activation layer

        # We chose to do either BatchNorm or dropout but not both (reference: https://arxiv.org/abs/1801.05134)
        layers = []

        activation_or_batchnorm = trial.suggest_categorical('activation_or_batchnorm_layer',
                                                            ['activation', 'BatchNorm_1'])

        if activation_or_batchnorm == 'activation':
            layers.append(nn.ReLU())

            optional_regularization_layer = trial.suggest_categorical('regularization_layer',
                                                                      ['dropout', 'BatchNorm_2', None])

            if optional_regularization_layer == 'dropout':
                drop_out = trial.suggest_uniform('drop_out_value', 0.1, 0.5)
                layers.append(nn.Dropout(p=drop_out))

            elif optional_regularization_layer == 'BatchNorm_2':
                layers.append(nn.BatchNorm1d(hidden_dim))

        elif activation_or_batchnorm == 'BatchNorm_1':
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())

        # We add one or zero extra linear layers after applying either dropout, batchnorm or activation layer
        n_linear_layers = trial.suggest_int("n_linear_layers", 0, 1)

        if (n_linear_layers != 0) & (len(layers) != 0):
            out_features = trial.suggest_int("n_units_l{}".format(int), 10, 20)

            layers.append(nn.Linear(hidden_dim, out_features))
            layers.append(nn.ReLU())

        params.update({'optional_model': nn.Sequential(*layers)})

        return nn.Sequential(*layers), params

    def calculate_auc(self, trial):
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'layer_dim': trial.suggest_int('layer_dim', 1, 3),
            'hidden_dim_seq': trial.suggest_int('hidden_dim_seq', 5, 15, 1),
            'hidden_dim_static': trial.suggest_int('hidden_dim_static', 15, 25, 1),
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
            'num_epochs': 6,
            'batch_size': trial.suggest_int('batch_size', 10, 60, 10),
            'feature_name': self.feature_name
        }

        if params['layer_dim'] == 1:
            params['drop_out_lstm'] = 0.0

        else:
            params['drop_out_lstm'] = trial.suggest_uniform('drop_out_lstm', 0.1, 0.5)

        train_loader_list = generate_dataloader(self.x_train, self.x_train_processed, self.y_train_processed,
                                                self.features_to_use, self.features_to_use_static,
                                                self.rec_ids_train_inner, FLAGS.reduced_seq_length,
                                                FLAGS.sub_seq_length, self.num_sub_sequences,
                                                params['batch_size'], test_phase=False)

        test_loader_list, rec_ids_list = generate_dataloader(self.x_test, self.x_test_processed, self.y_test_processed,
                                                             self.features_to_use, self.features_to_use_static,
                                                             self.rec_ids_test_inner, FLAGS.reduced_seq_length,
                                                             FLAGS.sub_seq_length, self.num_sub_sequences,
                                                             params['batch_size'], test_phase=True)

        n_classes = 1
        input_dim_seq = len(self.features_to_use)
        input_dim_static = len(self.features_to_use_static)
        device = 'cpu'

        optional_model_part, params = self.build_model_custom(trial,
                                                              hidden_dim=(params['hidden_dim_seq'] + params['hidden_dim_static']),
                                                              params=params)
        optional_model_part.to(device)

        model_lstm_combined = LSTMCombinedModel(input_dim_seq=input_dim_seq, hidden_dim_seq=params['hidden_dim_seq'],
                                                input_dim_static=input_dim_static,
                                                hidden_dim_static=params['hidden_dim_static'],
                                                layer_dim=params['layer_dim'], bidirectional=params['bidirectional'],
                                                batch_size=params['batch_size'], output_dim=n_classes,
                                                model_optional=optional_model_part,
                                                dropout_prob=params['drop_out_lstm'], device=device)

        model_lstm_combined.to(device)

        # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
        # https://discuss.pytorch.org/t/bcewithlogitsloss-and-class-weights/88837
        pos_weight = torch.Tensor([self.pos_weight])

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(model_lstm_combined.parameters(), lr=params['learning_rate'])

        opt_lstm = OptimizationCombinedLSTM(model=model_lstm_combined, loss_fn=loss_fn, optimizer=optimizer,
                                            num_sub_sequences=self.num_sub_sequences, device=device)

        start = timer()

        auc_mean_prob, auc_max_prob, params = opt_lstm.train_combined_model(trial, train_loader_list, test_loader_list, self.features_to_use,
                                                     params, n_epochs=params['num_epochs'])

        run_time = timer() - start

        # Write to the csv file ('a' means append)
        of_connection = open(self.out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([auc_mean_prob, auc_max_prob, params, self.outer_fold, self.inner_fold, self.rec_ids_train_outer,
                         self.rec_ids_test_outer, self.rec_ids_train_inner, self.rec_ids_test_inner, run_time])
        of_connection.close()

        return auc_mean_prob, auc_max_prob


class ObjectiveTcnFeatureCombinedModelWithCopies(object):
    def __init__(self, trial, x_train, x_test, x_train_processed, x_test_processed, y_train_processed, y_test_processed,
                 rec_ids_train_inner, rec_ids_test_inner, rec_ids_train_outer, rec_ids_test_outer, pos_weight,
                 feature_name, num_sub_sequences, features_to_use, features_to_use_static, out_file,
                 outer_fold, inner_fold):
        self.x_train = x_train
        self.x_test = x_test
        self.x_train_processed = x_train_processed
        self.x_test_processed = x_test_processed
        self.y_train_processed = y_train_processed
        self.y_test_processed = y_test_processed
        self.rec_ids_train_inner = rec_ids_train_inner
        self.rec_ids_test_inner = rec_ids_test_inner
        self.rec_ids_train_outer = rec_ids_train_outer
        self.rec_ids_test_outer = rec_ids_test_outer
        self.pos_weight = pos_weight
        self.feature_name = feature_name
        self.num_sub_sequences = num_sub_sequences
        self.features_to_use = features_to_use
        self.features_to_use_static = features_to_use_static
        self.out_file = out_file
        self.num_static_features = len(self.features_to_use_static)
        self.outer_fold = outer_fold
        self.inner_fold = inner_fold

        self.loss = self.calculate_loss(trial)

    # Build a model by implementing define-by-run design from Optuna
    def build_model_custom(self, trial, hidden_dim, params):
        # These are the possible optional models we can construct here:

        # The last part of the model before building this optional part was a linear layer and
        # these are the options to go from there:

        # 1. Linear layer -> BatchNorm -> activation layer -> Linear Layer -> activation layer
        # 2. Linear layer -> BatchNorm -> activation layer
        # 3. Linear layer -> activation layer -> BatchNorm -> Linear Layer -> activation layer
        # 4. Linear layer -> activation layer -> BatchNorm
        # 5. Linear layer -> activation layer -> dropout -> Linear Layer -> activation layer
        # 6. Linear layer -> activation layer -> dropout
        # 7. Linear layer -> activation layer

        # We chose to do either BatchNorm or dropout but not both (reference: https://arxiv.org/abs/1801.05134)
        layers = []

        activation_or_batchnorm = trial.suggest_categorical('activation_or_batchnorm_layer',
                                                            ['activation', 'BatchNorm_1'])

        if activation_or_batchnorm == 'activation':
            layers.append(nn.ReLU())

            optional_regularization_layer = trial.suggest_categorical('regularization_layer',
                                                                      ['dropout', 'BatchNorm_2', None])

            if optional_regularization_layer == 'dropout':
                drop_out = trial.suggest_uniform('drop_out_value', 0.1, 0.5)
                layers.append(nn.Dropout(p=drop_out))

            elif optional_regularization_layer == 'BatchNorm_2':
                layers.append(nn.BatchNorm1d(hidden_dim))

        elif activation_or_batchnorm == 'BatchNorm_1':
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())

        # We add one or zero extra linear layers after applying either dropout, batchnorm or activation layer
        n_linear_layers = trial.suggest_int("n_linear_layers", 0, 1)

        if (n_linear_layers != 0) & (len(layers) != 0):
            out_features = trial.suggest_int("n_units_l{}".format(int), 10, 20)

            layers.append(nn.Linear(hidden_dim, out_features))
            layers.append(nn.ReLU())

        params.update({'optional_model': nn.Sequential(*layers)})

        return nn.Sequential(*layers), params

    def calculate_loss(self, trial):
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'num_hidden_units_per_layer': trial.suggest_int('num_hidden_units_per_layer', 25, 35, 1),
            'kernel_size': trial.suggest_int('kernel_size', 3, 9, 2),
            'num_epochs': 6,
            'drop_out': trial.suggest_uniform('drop_out', 0.1, 0.5),
            'batch_size': trial.suggest_int('batch_size', 10, 60, 10),
            'feature_name': self.feature_name
        }

        train_loader_list = generate_dataloader(self.x_train, self.x_train_processed, self.y_train_processed,
                                                self.features_to_use, self.features_to_use_static,
                                                self.rec_ids_train_inner, FLAGS.reduced_seq_length,
                                                FLAGS.sub_seq_length, self.num_sub_sequences,
                                                params['batch_size'], test_phase=False)

        test_loader_list, rec_ids_list = generate_dataloader(self.x_test, self.x_test_processed, self.y_test_processed,
                                                             self.features_to_use, self.features_to_use_static,
                                                             self.rec_ids_test_inner, FLAGS.reduced_seq_length,
                                                             FLAGS.sub_seq_length, self.num_sub_sequences,
                                                             params['batch_size'], test_phase=True)

        n_classes = 1
        input_channels = len(self.features_to_use) + self.num_static_features
        device = 'cpu'

        params['num_levels'] = get_num_levels_based_on_receptive_field(kernel_size=params['kernel_size'],
                                                                       receptive_field=FLAGS.sub_seq_length,
                                                                       dilation_exponential_base=2)

        channel_sizes = [params['num_hidden_units_per_layer']] * params['num_levels']

        # We fixate the stride at 1
        params['stride'] = 1
        optional_model_part, params = self.build_model_custom(trial,
                                                              hidden_dim=params['num_hidden_units_per_layer'],
                                                              params=params)
        optional_model_part.to(device)

        model_tcn_combined = TCNCombinedModelCopies(input_channels, n_classes, channel_sizes, stride=params['stride'],
                                                    kernel_size=params['kernel_size'], dropout=params['drop_out'],
                                                    hidden_dim_combined=params['num_hidden_units_per_layer'],
                                                    model_optional=optional_model_part)
        model_tcn_combined.to(device)

        # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
        # https://discuss.pytorch.org/t/bcewithlogitsloss-and-class-weights/88837
        pos_weight = torch.Tensor([self.pos_weight])

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(model_tcn_combined.parameters(), lr=params['learning_rate'])

        opt_tcn = OptimizationTCNFeatureSequenceCombinedCopies(model=model_tcn_combined, loss_fn=loss_fn,
                                                               optimizer=optimizer,
                                                               num_sub_sequences=self.num_sub_sequences,
                                                               device=device)

        # Keep track of evals
        start = timer()
        loss, params = opt_tcn.train(trial, train_loader_list, test_loader_list, params, n_epochs=params['num_epochs'])

        run_time = timer() - start

        # Write to the csv file ('a' means append)
        of_connection = open(self.out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([loss, params, self.outer_fold, self.inner_fold, self.rec_ids_train_outer,
                         self.rec_ids_test_outer, self.rec_ids_train_inner, self.rec_ids_test_inner, run_time])
        of_connection.close()

        return loss


class ObjectiveTcnFeatureCombinedModel(object):
    def __init__(self, trial, x_train, x_test, x_train_processed, x_test_processed, y_train_processed,
                 y_test_processed, rec_ids_train_inner, rec_ids_test_inner, rec_ids_train_outer, rec_ids_test_outer,
                 pos_weight, feature_name, num_sub_sequences, features_to_use, features_to_use_static, out_file,
                 outer_fold, inner_fold):
        self.x_train = x_train
        self.x_test = x_test
        self.x_train_processed = x_train_processed
        self.x_test_processed = x_test_processed
        self.y_train_processed = y_train_processed
        self.y_test_processed = y_test_processed
        self.rec_ids_train_inner = rec_ids_train_inner
        self.rec_ids_test_inner = rec_ids_test_inner
        self.rec_ids_train_outer = rec_ids_train_outer
        self.rec_ids_test_outer = rec_ids_test_outer
        self.pos_weight = pos_weight
        self.feature_name = feature_name
        self.num_sub_sequences = num_sub_sequences
        self.features_to_use = features_to_use
        self.features_to_use_static = features_to_use_static
        self.out_file = out_file
        self.num_static_features = len(self.features_to_use_static)
        self.outer_fold = outer_fold
        self.inner_fold = inner_fold

        self.auc = self.calculate_auc(trial)

    # Build a model by implementing define-by-run design from Optuna
    def build_model_custom(self, trial, hidden_dim, params):
        # These are the possible optional models we can construct here:

        # The last part of the model before building this optional part was a linear layer and
        # these are the options to go from there:

        # 1. Linear layer -> BatchNorm -> activation layer -> Linear Layer -> activation layer
        # 2. Linear layer -> BatchNorm -> activation layer
        # 3. Linear layer -> activation layer -> BatchNorm -> Linear Layer -> activation layer
        # 4. Linear layer -> activation layer -> BatchNorm
        # 5. Linear layer -> activation layer -> dropout -> Linear Layer -> activation layer
        # 6. Linear layer -> activation layer -> dropout
        # 7. Linear layer -> activation layer

        # We chose to do either BatchNorm or dropout but not both (reference: https://arxiv.org/abs/1801.05134)
        layers = []

        activation_or_batchnorm = trial.suggest_categorical('activation_or_batchnorm_layer',
                                                            ['activation', 'BatchNorm_1'])

        if activation_or_batchnorm == 'activation':
            layers.append(nn.ReLU())

            optional_regularization_layer = trial.suggest_categorical('regularization_layer',
                                                                      ['dropout', 'BatchNorm_2', None])

            if optional_regularization_layer == 'dropout':
                drop_out = trial.suggest_uniform('drop_out_value', 0.1, 0.5)
                layers.append(nn.Dropout(p=drop_out))

            elif optional_regularization_layer == 'BatchNorm_2':
                layers.append(nn.BatchNorm1d(hidden_dim))

        elif activation_or_batchnorm == 'BatchNorm_1':
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())

        # We add one or zero extra linear layers after applying either dropout, batchnorm or activation layer
        n_linear_layers = trial.suggest_int("n_linear_layers", 0, 1)

        if (n_linear_layers != 0) & (len(layers) != 0):
            out_features = trial.suggest_int("n_units_l{}".format(int), 10, 20)

            layers.append(nn.Linear(hidden_dim, out_features))
            layers.append(nn.ReLU())

        params.update({'optional_model': nn.Sequential(*layers)})

        return nn.Sequential(*layers), params

    def calculate_auc(self, trial):
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'num_hidden_units_per_layer': trial.suggest_int('num_hidden_units_per_layer', 5, 15, 1),
            'kernel_size': trial.suggest_int('kernel_size', 3, 9, 2),
            'num_epochs': 6,
            'drop_out': trial.suggest_uniform('drop_out', 0.1, 0.5),
            'batch_size': trial.suggest_int('batch_size', 10, 60, 10),
            'hidden_dim_static': trial.suggest_int('hidden_dim_static', 15, 25, 1),
            'feature_name': self.feature_name
        }

        print(f'Params: {params}')

        train_loader_list = generate_dataloader(self.x_train, self.x_train_processed, self.y_train_processed,
                                                self.features_to_use, self.features_to_use_static,
                                                self.rec_ids_train_inner, FLAGS.reduced_seq_length,
                                                FLAGS.sub_seq_length, self.num_sub_sequences,
                                                params['batch_size'], test_phase=False)

        test_loader_list, rec_ids_list = generate_dataloader(self.x_test, self.x_test_processed, self.y_test_processed,
                                                             self.features_to_use, self.features_to_use_static,
                                                             self.rec_ids_test_inner, FLAGS.reduced_seq_length,
                                                             FLAGS.sub_seq_length, self.num_sub_sequences,
                                                             params['batch_size'], test_phase=True)

        n_classes = 1
        input_channels = len(self.features_to_use)
        input_dim_static = self.num_static_features
        device = 'cpu'

        params['num_levels'] = get_num_levels_based_on_receptive_field(kernel_size=params['kernel_size'],
                                                                       receptive_field=FLAGS.sub_seq_length,
                                                                       dilation_exponential_base=2)

        channel_sizes = [params['num_hidden_units_per_layer']] * params['num_levels']

        # We fixate the stride at 1
        params['stride'] = 1
        optional_model_part, params = self.build_model_custom(trial,
                                                              hidden_dim=params['num_hidden_units_per_layer'] + params['hidden_dim_static'],
                                                              params=params)
        optional_model_part.to(device)

        model_tcn_combined = TCNCombinedModel(input_channels, n_classes, channel_sizes, stride=params['stride'],
                                              kernel_size=params['kernel_size'], dropout=params['drop_out'],
                                              input_dim_static=input_dim_static,
                                              hidden_dim_combined=params['num_hidden_units_per_layer'] + params['hidden_dim_static'],
                                              model_optional=optional_model_part)
        model_tcn_combined.to(device)

        # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
        # https://discuss.pytorch.org/t/bcewithlogitsloss-and-class-weights/88837
        pos_weight = torch.Tensor([self.pos_weight])

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(model_tcn_combined.parameters(), lr=params['learning_rate'])

        opt_tcn = OptimizationTCNFeatureSequenceCombined(model=model_tcn_combined, loss_fn=loss_fn, optimizer=optimizer,
                                                         num_sub_sequences=self.num_sub_sequences, device=device)

        # Keep track of evals
        start = timer()
        auc_mean_prob, auc_max_prob, params = opt_tcn.train(trial, train_loader_list, test_loader_list, features_to_use,
                                                            params, n_epochs=params['num_epochs'])

        run_time = timer() - start

        # Write to the csv file ('a' means append)
        of_connection = open(self.out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([auc_mean_prob, auc_max_prob, params, self.outer_fold, self.inner_fold, self.rec_ids_train_outer,
                         self.rec_ids_test_outer, self.rec_ids_train_inner, self.rec_ids_test_inner, run_time])
        of_connection.close()

        return auc_mean_prob, auc_max_prob


class ObjectiveLSTMFeatureModel(object):
    def __init__(self, trial, x_train, x_test, x_train_processed, x_test_processed, y_train_processed, y_test_processed,
                 rec_ids_train_inner, rec_ids_test_inner, rec_ids_train_outer, rec_ids_test_outer, pos_weight,
                 feature_name, num_sub_sequences, features_to_use, features_to_use_static, out_file,
                 outer_fold, inner_fold):
        self.x_train = x_train
        self.x_test = x_test
        self.x_train_processed = x_train_processed
        self.x_test_processed = x_test_processed
        self.y_train_processed = y_train_processed
        self.y_test_processed = y_test_processed
        self.rec_ids_train_inner = rec_ids_train_inner
        self.rec_ids_test_inner = rec_ids_test_inner
        self.rec_ids_train_outer = rec_ids_train_outer
        self.rec_ids_test_outer = rec_ids_test_outer
        self.pos_weight = pos_weight
        self.feature_name = feature_name
        self.num_sub_sequences = num_sub_sequences
        self.features_to_use = features_to_use
        self.features_to_use_static = features_to_use_static
        self.out_file = out_file
        self.outer_fold = outer_fold
        self.inner_fold = inner_fold

        self.auc = self.calculate_auc(trial)

    def calculate_auc(self, trial):

        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'layer_dim': trial.suggest_int('layer_dim', 1, 3),
            'hidden_dim': trial.suggest_int('hidden_dim', 5, 15, 1),
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
            'num_epochs': 3,
            'batch_size': trial.suggest_int('batch_size', 10, 60, 10),
            'feature_name': self.feature_name
        }

        if params['layer_dim'] == 1:
            params['drop_out_lstm'] = 0.0

        else:
            params['drop_out_lstm'] = trial.suggest_uniform('drop_out_lstm', 0.1, 0.5)

        train_loader_list = generate_dataloader(self.x_train, self.x_train_processed, self.y_train_processed,
                                                self.features_to_use, self.features_to_use_static, self.rec_ids_train_inner,
                                                FLAGS.reduced_seq_length, FLAGS.sub_seq_length, self.num_sub_sequences,
                                                params['batch_size'], test_phase=False)

        test_loader_list, rec_ids_list = generate_dataloader(self.x_test, self.x_test_processed, self.y_test_processed,
                                                             self.features_to_use, self.features_to_use_static,
                                                             self.rec_ids_test_inner, FLAGS.reduced_seq_length,
                                                             FLAGS.sub_seq_length, self.num_sub_sequences,
                                                             params['batch_size'], test_phase=True)

        n_classes = 1
        input_channels = len(self.features_to_use)
        # The num_sub_sequences variable is the number of sub_sequences necessary to complete an entire sequence
        device = 'cpu'

        model_lstm_feature_stateful = LSTMStatefulClassificationFeatureSequence(input_size=input_channels,
                                                                                hidden_size=params['hidden_dim'],
                                                                                num_layers=params['layer_dim'],
                                                                                dropout=params['drop_out_lstm'],
                                                                                output_size=n_classes,
                                                                                bidirectional=params['bidirectional'],
                                                                                batch_size=params['batch_size'],
                                                                                device=device, batch_first=True)
        model_lstm_feature_stateful.to(device)

        # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
        # https://discuss.pytorch.org/t/bcewithlogitsloss-and-class-weights/88837
        pos_weight = torch.Tensor([self.pos_weight])

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(model_lstm_feature_stateful.parameters(),
                               lr=params['learning_rate'])

        opt_model_lstm_stateful_feature = OptimizationStatefulFeatureSequenceLSTM(model=model_lstm_feature_stateful,
                                                                                  loss_fn=loss_fn,
                                                                                  optimizer=optimizer,
                                                                                  num_sub_sequences=self.num_sub_sequences,
                                                                                  device=device)

        start = timer()

        auc_mean_prob, auc_max_prob, params = opt_model_lstm_stateful_feature.train(trial, train_loader_list,
                                                                                    test_loader_list, params,
                                                                                    n_epochs=params['num_epochs'])

        run_time = timer() - start

        # Write to the csv file ('a' means append)
        of_connection = open(self.out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([auc_mean_prob, auc_max_prob, params, self.outer_fold, self.inner_fold, self.rec_ids_train_outer,
                         self.rec_ids_test_outer, self.rec_ids_train_inner, self.rec_ids_test_inner, run_time])
        of_connection.close()

        return auc_mean_prob, auc_max_prob


class ObjectiveTcnFeatureModel(object):
    def __init__(self, trial, x_train, x_test, x_train_processed, x_test_processed, y_train_processed,
                 y_test_processed, rec_ids_train_inner, rec_ids_test_inner, rec_ids_train_outer, rec_ids_test_outer,
                 pos_weight, feature_name, num_sub_sequences, features_to_use, features_to_use_static, add_static_data,
                 out_file, outer_fold, inner_fold):
        self.x_train = x_train
        self.x_test = x_test
        self.x_train_processed = x_train_processed
        self.x_test_processed = x_test_processed
        self.y_train_processed = y_train_processed
        self.y_test_processed = y_test_processed
        self.rec_ids_train_inner = rec_ids_train_inner # unique rec ids of x train inner loop
        self.rec_ids_test_inner = rec_ids_test_inner # unique rec ids of x test inner loop
        self.rec_ids_train_outer = rec_ids_train_outer # unique rec ids of x train outer loop
        self.rec_ids_test_outer = rec_ids_test_outer # unique rec ids of x test outer loop
        self.pos_weight = pos_weight
        self.feature_name = feature_name
        self.num_sub_sequences = num_sub_sequences
        self.features_to_use = features_to_use
        self.features_to_use_static = features_to_use_static
        self.add_static_data = add_static_data
        self.out_file = out_file
        self.outer_fold = outer_fold
        self.inner_fold = inner_fold

        self.auc = self.calculate_auc(trial)

    def calculate_auc(self, trial):
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'num_hidden_units_per_layer': trial.suggest_int('num_hidden_units_per_layer', 5, 15, 1),
            'kernel_size': trial.suggest_int('kernel_size', 3, 9, 2),
            'num_epochs': 3,
            'drop_out': trial.suggest_uniform('drop_out', 0.1, 0.5),
            'batch_size': trial.suggest_int('batch_size', 10, 60, 10),
            'feature_name': self.feature_name
        }
        print(f'Params: {params}')

        train_loader_list = generate_dataloader(self.x_train, self.x_train_processed, self.y_train_processed,
                                                self.features_to_use, self.features_to_use_static,
                                                self.rec_ids_train_inner, FLAGS.reduced_seq_length,
                                                FLAGS.sub_seq_length, self.num_sub_sequences,
                                                params['batch_size'], test_phase=False)

        test_loader_list, rec_ids_list = generate_dataloader(self.x_test, self.x_test_processed, self.y_test_processed,
                                                             self.features_to_use, self.features_to_use_static,
                                                             self.rec_ids_test_inner, FLAGS.reduced_seq_length,
                                                             FLAGS.sub_seq_length, self.num_sub_sequences,
                                                             params['batch_size'], test_phase=True)

        n_classes = 1
        input_channels = len(self.features_to_use)
        device = 'cpu'

        params['num_levels'] = get_num_levels_based_on_receptive_field(kernel_size=params['kernel_size'],
                                                                       receptive_field=FLAGS.sub_seq_length,
                                                                       dilation_exponential_base=2)

        channel_sizes = [params['num_hidden_units_per_layer']] * params['num_levels']

        # For now we fixate the stride at 1
        params['stride'] = 1

        model_tcn = TCN(input_channels, n_classes, channel_sizes, stride=params['stride'],
                        kernel_size=params['kernel_size'], dropout=params['drop_out'])
        model_tcn.to(device)

        # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
        # https://discuss.pytorch.org/t/bcewithlogitsloss-and-class-weights/88837
        pos_weight = torch.Tensor([self.pos_weight])

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(model_tcn.parameters(), lr=params['learning_rate'])

        opt_tcn = OptimizationTCNFeatureSequence(model=model_tcn, loss_fn=loss_fn, optimizer=optimizer,
                                                 num_sub_sequences=self.num_sub_sequences, device=device)

        # Keep track of evals
        start = timer()

        auc_mean, auc_max, params = opt_tcn.train(trial, train_loader_list, test_loader_list, params, n_epochs=params['num_epochs'])

        run_time = timer() - start

        # Write to the csv file ('a' means append)
        of_connection = open(self.out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([auc_mean, auc_max, params, self.outer_fold, self.inner_fold, self.rec_ids_train_outer,
                         self.rec_ids_test_outer, self.rec_ids_train_inner, self.rec_ids_test_inner, run_time])
        of_connection.close()

        return auc_mean, auc_max


def objective_cv_inner(trial, X_train_outer_fold: pd.DataFrame, y_train_outer_fold,
                       X_train_static_outer_fold: pd.DataFrame, rec_ids_test_outer: List[int], out_file: str,
                       outer_fold_i: int, add_static_data: bool, model_name: str, feature_name: str,
                       df_ground_truth: pd.DataFrame):
    """Perform the inner loop of nested cross-validation for hyperoptimization. In this function
    the inner fold splits (3 folds) are created and hyperoptimization (optimized for both the auc on the mean
    prediction over all sub-sequences and max prediction over all sub-sequences) is done on these folds.

    The process of nested cross-validation is further explained here: https://weina.me/nested-cross-validation.
    """
    groups = np.array(X_train_outer_fold[c.REC_ID_NAME])
    rec_ids_x_train_outer_unique = list(X_train_outer_fold[c.REC_ID_NAME].unique())
    auc_mean_scores = []
    auc_max_scores = []

    skf_inner_groups = StratifiedGroupKFold(n_splits=3, random_state=0, shuffle=True)
    for inner_fold_j, (train_index_inner, test_index_inner) in enumerate(skf_inner_groups.split(X_train_outer_fold,
                                                                                                y_train_outer_fold,
                                                                                                groups)):

        X_train_signal_inner_fold = X_train_outer_fold.iloc[train_index_inner].copy().reset_index(drop=True)
        X_test_signal_inner_fold = X_train_outer_fold.iloc[test_index_inner].copy().reset_index(drop=True)
        y_train_inner_fold = y_train_outer_fold.iloc[train_index_inner].copy().reset_index(drop=True)
        y_test_inner_fold = y_train_outer_fold.iloc[test_index_inner].copy().reset_index(drop=True)

        pos_cases = y_train_inner_fold['premature'].value_counts()[1]
        neg_cases = y_train_inner_fold['premature'].value_counts()[0]
        pos_weight = neg_cases/pos_cases

        # We keep the rec ids order to later on merge the static data correctly
        rec_ids_x_train_inner_signal = list(X_train_signal_inner_fold[c.REC_ID_NAME])
        rec_ids_x_test_inner_signal = list(X_test_signal_inner_fold[c.REC_ID_NAME])

        # We keep the rec ids order to later on merge the static data correctly
        rec_ids_x_train_inner_unique = list(X_train_signal_inner_fold[c.REC_ID_NAME].unique())
        rec_ids_x_test_inner_unique = list(X_test_signal_inner_fold[c.REC_ID_NAME].unique())

        # The number of sub-sequences needed to make up an original sequence
        num_sub_sequences_fixed = int(FLAGS.reduced_seq_length / FLAGS.sub_seq_length)

        X_train_signal_inner_fold_processed, X_test_signal_inner_fold_processed, \
        y_train_inner_fold_processed, y_test_inner_fold_processed = preprocess_signal_data(X_train_signal_inner_fold,
                                                                                           X_test_signal_inner_fold,
                                                                                           y_train_inner_fold,
                                                                                           y_test_inner_fold,
                                                                                           features_to_use)

        label_check_train = df_ground_truth.loc[df_ground_truth[c.REC_ID_NAME].isin(rec_ids_x_train_inner_unique), 'premature'].copy().reset_index(drop=True)
        label_check_test = df_ground_truth.loc[df_ground_truth[c.REC_ID_NAME].isin(rec_ids_x_test_inner_unique), 'premature'].copy().reset_index(drop=True)

        assert all(x == y for x, y in zip(y_train_inner_fold_processed, label_check_train.values))
        assert all(x == y for x, y in zip(y_test_inner_fold_processed, label_check_test.values))

        if FLAGS.add_static_data:
            X_train_static_inner_fold = X_train_static_outer_fold.loc[
                    X_train_static_outer_fold[c.REC_ID_NAME].isin(rec_ids_x_train_inner_unique)].copy().reset_index(drop=True)
            X_test_static_inner_fold = X_train_static_outer_fold.loc[
                    X_train_static_outer_fold[c.REC_ID_NAME].isin(rec_ids_x_test_inner_unique)].copy().reset_index(drop=True)

            X_train_combined_inner_fold, X_test_combined_inner_fold, X_train_combined_processed, \
            X_test_combined_processed, selected_columns_train_static = add_static_data_to_signal_data(X_train_static_inner_fold,
                                                                                                      X_test_static_inner_fold,
                                                                                                      X_train_signal_inner_fold,
                                                                                                      X_test_signal_inner_fold,
                                                                                                      X_train_signal_inner_fold_processed,
                                                                                                      X_test_signal_inner_fold_processed,
                                                                                                      rec_ids_x_train_inner_signal,
                                                                                                      rec_ids_x_test_inner_signal,
                                                                                                      features_to_use,
                                                                                                      threshold_correlation=0.85)

            assert all(x == y for x, y in zip(X_train_combined_inner_fold[c.REC_ID_NAME].unique(), rec_ids_x_train_inner_unique))
            assert all(x == y for x, y in zip(X_test_combined_inner_fold[c.REC_ID_NAME].unique(), rec_ids_x_test_inner_unique))

        if model_name == 'tcn' and not add_static_data:
            features_to_use_static = []
            auc_mean, auc_max = ObjectiveTcnFeatureModel(trial, X_train_signal_inner_fold, X_test_signal_inner_fold,
                                                         X_train_signal_inner_fold_processed,
                                                         X_test_signal_inner_fold_processed,
                                                         y_train_inner_fold_processed, y_test_inner_fold_processed,
                                                         rec_ids_x_train_inner_unique, rec_ids_x_test_inner_unique,
                                                         rec_ids_x_train_outer_unique, rec_ids_test_outer, pos_weight,
                                                         feature_name, num_sub_sequences_fixed, features_to_use,
                                                         features_to_use_static, add_static_data, out_file,
                                                         outer_fold_i, inner_fold_j).auc

        if model_name == 'tcn' and FLAGS.add_static_data and not FLAGS.use_copies_for_static_data:
            auc_mean, auc_max = ObjectiveTcnFeatureCombinedModel(trial, X_train_combined_inner_fold,
                                                                 X_test_combined_inner_fold, X_train_combined_processed,
                                                                 X_test_combined_processed,
                                                                 y_train_inner_fold_processed,
                                                                 y_test_inner_fold_processed,
                                                                 rec_ids_x_train_inner_unique,
                                                                 rec_ids_x_test_inner_unique,
                                                                 rec_ids_x_train_outer_unique, rec_ids_test_outer,
                                                                 pos_weight, feature_name, num_sub_sequences_fixed,
                                                                 features_to_use, selected_columns_train_static,
                                                                 out_file, outer_fold_i, inner_fold_j).auc

        elif model_name == 'tcn' and FLAGS.add_static_data and FLAGS.use_copies_for_static_data:
            loss = ObjectiveTcnFeatureCombinedModelWithCopies(trial, X_train_combined_inner_fold,
                                                              X_test_combined_inner_fold, X_train_combined_processed,
                                                              X_test_combined_processed, y_train_inner_fold_processed,
                                                              y_test_inner_fold_processed,
                                                              rec_ids_x_train_inner_unique, rec_ids_x_test_inner_unique,
                                                              rec_ids_x_train_outer_unique, rec_ids_test_outer,
                                                              pos_weight, feature_name, num_sub_sequences_fixed,
                                                              features_to_use, selected_columns_train_static, out_file,
                                                              outer_fold_i, inner_fold_j).loss

        elif model_name == 'lstm' and not FLAGS.add_static_data:
            features_to_use_static = []

            auc_mean, auc_max = ObjectiveLSTMFeatureModel(trial, X_train_signal_inner_fold, X_test_signal_inner_fold,
                                                          X_train_signal_inner_fold_processed,
                                                          X_test_signal_inner_fold_processed,
                                                          y_train_inner_fold_processed, y_test_inner_fold_processed,
                                                          rec_ids_x_train_inner_unique, rec_ids_x_test_inner_unique,
                                                          rec_ids_x_train_outer_unique, rec_ids_test_outer, pos_weight,
                                                          feature_name, num_sub_sequences_fixed, features_to_use,
                                                          features_to_use_static, out_file, outer_fold_i,
                                                          inner_fold_j).auc

        elif model_name == 'lstm' and FLAGS.add_static_data:
            auc_mean, auc_max = ObjectiveLSTMFeatureCombinedModel(trial, X_train_combined_inner_fold,
                                                                  X_test_combined_inner_fold,
                                                                  X_train_combined_processed, X_test_combined_processed,
                                                                  y_train_inner_fold_processed,
                                                                  y_test_inner_fold_processed,
                                                                  rec_ids_x_train_inner_unique,
                                                                  rec_ids_x_test_inner_unique,
                                                                  rec_ids_x_train_outer_unique, rec_ids_test_outer,
                                                                  pos_weight, feature_name, num_sub_sequences_fixed,
                                                                  features_to_use, selected_columns_train_static,
                                                                  out_file, outer_fold_i, inner_fold_j).auc

        auc_mean_scores.append(auc_mean)
        auc_max_scores.append(auc_max)

    return np.mean(auc_mean_scores), np.mean(auc_max_scores)


def objective_cv_outer(trial):
    """Perform nested cross-validation for hyperoptimization. In this function the outer fold splits (5 folds) are
    created and the cross-validation function for the inner folds will be called. For each of the 5 folds,
    hyperoptimization will be performed within the inner loop, resulting in 5 distinct optimal hyperparameters (one for
    each outer fold).

    The process of nested cross-validation is further explained here: https://weina.me/nested-cross-validation
    """
    # Load dataset from hard disk, this is the filtered signal data
    df_signals_new = pd.read_csv(f'{data_path}/df_signals_filt.csv', sep=';')

    df_clinical_information = build_clinical_information_dataframe(data_path, settings_path)

    df_features, df_label = basic_preprocessing_signal_data(df_signals_new, df_clinical_information,
                                                            FLAGS.reduced_seq_length, features_to_use,
                                                            FLAGS.feature_name, fs)

    df_features = convert_columns_to_numeric(df_features, [c.REC_ID_NAME])
    df_features[c.REC_ID_NAME] = df_features[c.REC_ID_NAME].astype(np.int64)

    groups = np.array(df_features[c.REC_ID_NAME])

    ground_truth_rec_id_and_label = pd.concat([df_features[c.REC_ID_NAME], df_label], axis=1)

    if FLAGS.add_static_data:
        df_static_information = basic_preprocessing_static_data(data_path, settings_path, df_clinical_information)

    auc_mean_outer = []
    auc_max_outer = []

    skf_outer_groups = StratifiedGroupKFold(n_splits=5, random_state=0, shuffle=True)
    for outer_fold_i, (train_index_outer, test_index_outer) in enumerate(skf_outer_groups.split(df_features, df_label,
                                                                                                groups)):

        X_train_signal_outer_fold = df_features.iloc[train_index_outer].copy().reset_index(drop=True)
        X_test_signal_outer_fold = df_features.iloc[test_index_outer].copy().reset_index(drop=True)
        y_train_outer_fold = df_label.iloc[train_index_outer].copy().reset_index(drop=True)
        y_test_outer_fold = df_label.iloc[test_index_outer].copy().reset_index(drop=True)

        # We keep the rec ids order to later on merge the static data correctly
        rec_ids_x_train_outer_unique = list(X_train_signal_outer_fold[c.REC_ID_NAME].unique())
        rec_ids_x_test_outer_unique = list(X_test_signal_outer_fold[c.REC_ID_NAME].unique())

        if FLAGS.add_static_data:
            X_train_static_outer_fold = df_static_information.loc[df_static_information[c.REC_ID_NAME].isin(rec_ids_x_train_outer_unique)].copy().reset_index(drop=True)
            X_test_static_outer_fold = df_static_information.loc[df_static_information[c.REC_ID_NAME].isin(rec_ids_x_test_outer_unique)].copy().reset_index(drop=True)

            # Safety checks
            assert all(x == y for x, y in zip(X_train_signal_outer_fold[c.REC_ID_NAME].unique(),
                                              X_train_static_outer_fold[c.REC_ID_NAME].unique())), \
                "Rec ids in X_train_signal and X_train_static must be in the exact same order!"

            assert all(x == y for x, y in zip(X_test_signal_outer_fold[c.REC_ID_NAME].unique(),
                                              X_test_static_outer_fold[c.REC_ID_NAME].unique())), \
                "Rec ids in X_test_signal and X_test_static must be in the exact same order!"

            assert set(X_train_signal_outer_fold[c.REC_ID_NAME]) == set(X_train_static_outer_fold[c.REC_ID_NAME]), \
                "Rec ids in train signals and train static data are not the same!"

            assert set(X_test_signal_outer_fold[c.REC_ID_NAME]) == set(X_test_static_outer_fold[c.REC_ID_NAME]), \
                "Rec ids in test signals and test static data are not the same!"

        else:
            X_train_static_outer_fold = None

        auc_mean_scores, auc_max_scores = objective_cv_inner(trial, X_train_signal_outer_fold, y_train_outer_fold,
                                                             X_train_static_outer_fold, rec_ids_x_test_outer_unique,
                                                             out_file, outer_fold_i, FLAGS.add_static_data, FLAGS.model,
                                                             FLAGS.feature_name, ground_truth_rec_id_and_label)

        auc_mean_outer.append(auc_mean_scores)
        auc_max_outer.append(auc_max_scores)

    return np.mean(auc_mean_outer), np.mean(auc_max_outer)


def main(model_name: str, feature_name: str, study, add_static_data: bool, copies: bool,
         output_path: str, n_trials: int):

    # File to save results
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)

    # Write the headers to the file
    writer.writerow(['mean_prob_auc', 'max_prob_auc', 'params', 'outer_fold', 'inner_fold',  'rec_ids_train_outer',
                     'rec_ids_test_outer', 'rec_ids_train_inner', 'rec_ids_test_inner', 'train_time'])
    of_connection.close()

    study.optimize(objective_cv_outer, n_trials=n_trials)
    print(study.best_trials)

    if add_static_data and not copies:
        joblib.dump(study,
                    f'{output_path}/hyper_opt_{model_name}_feature_{feature_name}_combined_{current_date_and_time}.pkl')

    if add_static_data and copies:
        joblib.dump(study,
                    f'{output_path}/hyper_opt_{model_name}_feature_{feature_name}_combined_copies_{current_date_and_time}.pkl')

    if not add_static_data:
        joblib.dump(study, f'{output_path}/hyper_opt_{model_name}_feature_{feature_name}_{current_date_and_time}.pkl')


if __name__ == "__main__":
    # EHG data to use for modelling
    features_to_use = ['channel_1_filt_0.34_1_hz', 'channel_2_filt_0.34_1_hz', 'channel_3_filt_0.34_1_hz']
    fs = 20
    output_path = os.path.join(file_paths['output_path'], 'model/hyper_parameter_opt/')

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # Command line arguments
    parser = argparse.ArgumentParser(description='Hyperoptimization based on Bayesian Optimization using the Optuna '
                                                 'package. Hyperoptimization is done through a nested cross-validation '
                                                 'method. Hyperparameter spaces are defined in the '
                                                 'ObjectiveLSTMFeatureCombinedModel, '
                                                 'ObjectiveTcnFeatureCombinedModelWithCopies, '
                                                 'ObjectiveTcnFeatureCombinedModel, ObjectiveTcnFeatureModel and '
                                                 'ObjectiveLSTMFeatureModel classes. The output path where the results '
                                                 'will be saved needs to be defined in this main function.')

    parser.add_argument('--model', type=str, required=True, help="Select what model to use: 'lstm' or 'tcn'",
                        choices=['tcn', 'lstm'])

    parser.add_argument('--feature_name', type=str, required=True,
                        help="Select what feature to use for data reduction: 'sample_entropy', 'peak_frequency' or "
                             "'median_frequency'", choices=['sample_entropy', 'peak_frequency', 'median_frequency'])

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

    parser.add_argument('--new_study', action='store_true', required=('--existing_study' not in sys.argv),
                        help="Use this flag if you want to create a new study to do hyperparameter optimization. "
                             "Use either the --new_study or --existing_study flag.")
    parser.add_argument('--existing_study', dest='new_study', action='store_false',
                        help="Use this flag if you want to continue with a previously run study. You should also "
                             "specify --study_name 'name_of_your_study_file' when using the --existing_study flag."
                             "Use either the --new_study or --existing_study flag.")
    parser.set_defaults(new_study=True)

    parser.add_argument('--study_name', required=('--existing_study' in sys.argv), type=str,
                        help="Provide the name of the file that contains the previously run optimization. "
                             "Must be a .pkl file. Usage: --study_name 'name_of_your_study_file.pkl'")
    parser.add_argument('--n_trials', type=int, required=True, default=50,
                        help="Number of runs you want to do for hyperoptimization. Default is 50 runs.")

    FLAGS, _ = parser.parse_known_args()
    print(FLAGS)

    # If new study
    if FLAGS.new_study:
        study = optuna.create_study(directions=["maximize", "maximize"], sampler=optuna.samplers.TPESampler())
    # If continue with study
    if not FLAGS.new_study:
        study = joblib.load(f"{output_path}/{FLAGS.study_name}")

    current_date_and_time = "{:%Y-%m-%d_%H-%M}".format(datetime.datetime.now())

    if FLAGS.add_static_data:
        if not FLAGS.use_copies_for_static_data:
            out_file = f'{output_path}/{FLAGS.model}_data_trials_feature_{FLAGS.feature_name}_combined_{current_date_and_time}.csv'
        if FLAGS.use_copies_for_static_data:
            out_file = f'{output_path}/{FLAGS.model}_data_trials_feature_{FLAGS.feature_name}_combined_copies_{current_date_and_time}.csv'

    if not FLAGS.add_static_data:
        out_file = f'{output_path}/{FLAGS.model}_data_trials_feature_{FLAGS.feature_name}_{current_date_and_time}.csv'

    main(FLAGS.model, FLAGS.feature_name, study, FLAGS.add_static_data, FLAGS.use_copies_for_static_data,
         output_path, FLAGS.n_trials)
