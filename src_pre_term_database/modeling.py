import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import math
from utils import read_settings

settings_path = '/Users/AFischer/PycharmProjects/cocoon-project/references/settings'
file_paths = read_settings(settings_path, 'file_paths')


class SaveBestModel:
    """
    Save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, output_path, file_name_output):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, f'{output_path}/{file_name_output}.pth')


class LSTMModel(nn.Module):
    """
    Source for set-up:
    https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

    Parameters
    ----------
    input_dim : int
        Input dimension of LSTM model. (Number of features you want to include).
    hidden_dim : int
        Hidden size is number of features of the hidden state for LSTM.
        If you increase hidden size then you compute bigger feature as hidden state output.
    layer_dim : int
        Multiple LSTM units which contain hidden states with given hidden size.
        If num_layers=2 it would mean stacking two LSTMs together to form a stacked LSTM,
        with the second LSTM taking in outputs of the first LSTM and computing the final results
    bidirectional : Boolean
        True if bidirectional, False if not. Default is False.
    output_dim : int
        Number of classes you want to predict.
    dropout_prob : float
        Regularization parameter to probabilistically exclude input and recurrent connections to LSTM units
        from activation and weight updates while training a network.
    device : str
        'cuda' or 'cpu'
    """
    def __init__(self, input_dim, hidden_dim, layer_dim, bidirectional, output_dim, dropout_prob, device):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.device = device

        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1

        # LSTM layers
        # batch first: If True, then the input and output tensors are provided as [batch, seq_len, n_features]
        # instead of [seq_len, batch, n_features]. Note that this does not apply to hidden or cell states.
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True,
                            bidirectional=self.bidirectional, dropout=dropout_prob)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim * num_directions, self.output_dim)

        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        Taken from: https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        To set forget-gate bias to 1: https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    # Source: http://proceedings.mlr.press/v37/jozefowicz15.pdf
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        num_directions = 2 if self.bidirectional else 1
        # Each time the forward pass is applied, the hidden cells and states are initialized on zeros. Meaning
        # that if you call model(x) in each mini-batch, the states and cells are initialized on zeros. We use
        # this for our case when the batches are independent of each other (a stateless LSTM model)
        h0 = torch.zeros(self.layer_dim * num_directions, x.size(0), self.hidden_dim,
                         device=self.device).requires_grad_()
        c0 = torch.zeros(self.layer_dim * num_directions, x.size(0), self.hidden_dim,
                         device=self.device).requires_grad_()

        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (h_t, c_t) = self.lstm(x, (h0.detach(), c0.detach()))

        # When you would use out[-1] in case of a bidirectional LSTM, you basically loose all
        # the information from the backward pass since you use the hidden state after only
        # one time step and not the entire sequence.

        # Sources for correctly handling the hidden states in case of bidirectional LSTM:
        # https://discuss.pytorch.org/t/bidirectional-3-layer-lstm-hidden-output/41336/5
        # https://discuss.pytorch.org/t/bilstm-output-hidden-state-mismatch/49825/2
        # https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/models/text/classifier/rnn.py
        # https://discuss.pytorch.org/t/rnn-output-vs-hidden-state-dont-match-up-my-misunderstanding/43280

        # Extract last hidden state
        # Use view to separate out the layers and directions first before adding them together.
        # We only want to know the hidden state of the last time step, as we're only interested in computing
        # the loss after the entire sequence length has been processed.
        last_layer_hidden_state = h_t.view(self.layer_dim, num_directions, x.size(0), self.hidden_dim)[-1]

        # Handle directions
        final_hidden_state = None
        if num_directions == 1:
            final_hidden_state = last_layer_hidden_state.squeeze()
        elif num_directions == 2:
            # h_1 and h_2 represent the last hidden states for the forward and backward pass in case
            # of a bidirectional LSTM.
            h_1, h_2 = last_layer_hidden_state[0], last_layer_hidden_state[1]
            final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states

        output = self.fc(final_hidden_state)

        return output


class LSTMCombinedModel(nn.Module):
    """ Model that combines both sequential and static data. Sequential data is processed according to a
    stateful LSTM model.

    Parameters
    ----------
    input_dim_seq : int
        Input dimension of the time series for the LSTM model. (Number of features you want to include).
    input_dim_static : int
        Input dimension of the static data. (Number of features you want to include).
    hidden_dim_seq : int
        Hidden size is number of features of the hidden state for LSTM (time series data).
        If you increase hidden size then you compute bigger feature as hidden state output.
    hidden_dim_static : int
        Hidden size is number of features of the hidden state for the static data.
        If you increase hidden size then you compute bigger feature as hidden state output.
    layer_dim : int
        Multiple LSTM units which contain hidden states with given hidden size.
        If num_layers=2 it would mean stacking two LSTMs together to form a stacked LSTM,
        with the second LSTM taking in outputs of the first LSTM and computing the final results
    bidirectional : Boolean
        True or False for a bidirectional LSTM.
    output_dim : int
        Number of classes you want to predict.
    model_optional : nn.Sequential
        Optional part of the model to process the sequential + static data through a couple of
        layers which have to be defined in a nn.Sequential object.
    dropout_prob : float
        Regularization parameter to probabilistically exclude input and recurrent connections to LSTM units
        from activation and weight updates while training a network.
    device : str
        'cuda' or 'cpu'.
    """
    def __init__(self, input_dim_seq, hidden_dim_seq, input_dim_static, hidden_dim_static, layer_dim,
                 bidirectional, batch_size, output_dim, model_optional, dropout_prob, device):
        super().__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim_seq = hidden_dim_seq
        self.hidden_dim_static = hidden_dim_static
        self.hidden_dim_combined = self.hidden_dim_seq + self.hidden_dim_static
        self.layer_dim = layer_dim
        self.bidirectional = bidirectional
        self.output_dim = output_dim

        num_directions = 2 if bidirectional else 1

        self.device = device

        # LSTM layers
        # batch first: If True, then the input and output tensors are provided as [batch, seq_len, n_features]
        # instead of [seq_len, batch, n_features]. Note that this does not apply to hidden or cell states.
        self.lstm = LSTMStateful(input_size=input_dim_seq, hidden_size=hidden_dim_seq,
                                 num_layers=layer_dim,
                                 bidirectional=self.bidirectional,
                                 batch_size=batch_size, device=device,
                                 batch_first=True,
                                 dropout=dropout_prob)

        self.model_optional = model_optional

        # Fully connected layer for the combined data
        self.fc_combined = nn.Linear((self.hidden_dim_seq * num_directions) + input_dim_static,
                                     self.hidden_dim_combined)

        for name, layer in self.model_optional.named_modules():
            if not isinstance(layer, nn.Linear):
                self.fc_output = nn.Linear(self.hidden_dim_combined, self.output_dim)

            if isinstance(layer, nn.Linear):
                self.fc_output = nn.Linear(layer.weight.size(dim=0), self.output_dim)
                break

        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        Taken from: https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        To set forget-gate bias to 1: https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        """
        for name, p in self.named_parameters():
            try:
                if 'lstm' in name:
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(p.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(p.data)
                    elif 'bias_ih' in name:
                        p.data.fill_(0)
                        # Set forget-gate bias to 1 for learning long-term dependencies
                        # Source: http://proceedings.mlr.press/v37/jozefowicz15.pdf
                        n = p.size(0)
                        p.data[(n // 4):(n // 2)].fill_(1)
                    elif 'bias_hh' in name:
                        p.data.fill_(0)
                elif 'fc' or 'model_optional' in name:
                    if 'weight' in name:
                        nn.init.xavier_uniform_(p.data)
                    elif 'bias' in name:
                        p.data.fill_(0)

            # This ValueError is for the case when model_optional is empty (i.e., nn.Sequential())
            except ValueError as e:
                if str(e) == "ValueError: Fan in and fan out can not be computed for tensor with fewer than 2 dimensions":
                    break

    def forward(self, x_seq, x_static):
        num_directions = 2 if self.bidirectional else 1

        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (h_t, c_t) = self.lstm(x_seq)

        # Extract last hidden state
        # Use view to separate out the layers and directions first before adding them together.
        # We only want to know the hidden state of the last time step, as we're only interested in computing
        # the loss after the entire sequence length has been processed.
        last_layer_hidden_state = h_t.view(self.layer_dim, num_directions, x_seq.size(0), self.hidden_dim_seq)[-1]

        # Handle directions
        final_hidden_state_seq = None
        if num_directions == 1:
            final_hidden_state_seq = last_layer_hidden_state.squeeze()
        elif num_directions == 2:
            # h_1 and h_2 represent the last hidden states for the forward and backward pass in case
            # of a bidirectional LSTM.
            h_1, h_2 = last_layer_hidden_state[0], last_layer_hidden_state[1]
            final_hidden_state_seq = torch.cat((h_1, h_2), 1)  # Concatenate both states

        # Here we add the static data to the processed (by the LSTM) time series data
        x_combined = torch.concat([final_hidden_state_seq, x_static], dim=1)

        # Through the linear layer
        x_combined = self.fc_combined(x_combined)

        # Optional model part
        x_combined = self.model_optional(x_combined)

        # Final linear layer
        output_combined = self.fc_output(x_combined)

        return output_combined


class LSTMStateful(nn.Module):
    """Create a stateful LSTM model for where the current hidden states and values
      are passed on to the next batch in the forward pass.

      When initializing an object of this class, the hidden states and cells will be resetted at value zero.

      Source: https://discuss.pytorch.org/t/lstm-stateful-batch-size-1/81002

      Parameters
      ----------
      input_size : int
          Input dimension of LSTM model.
      hidden_size : int
          Hidden states dimensionality.
      num_layers : int
        If num_layers > 1 then it is a stacked LSTM.
      bidirectional : bool
        Bidirectional LSTM or not.
      batch_size : int
          Number of samples in a batch.
      device : str
          'cuda' or 'cpu'
      batch_first : bool
        Default is True. If True, then the shape of the tensor that is fed to the LSTM is of shape
         [batch, seq_length, num_features].
      dropout : float
        Only applicable if num_layers > 1.
    """
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, batch_size,
                 device, batch_first, dropout):
        super().__init__()
        self.hidden_state, self.hidden_cell = (None, None)
        self.hidden_size = hidden_size
        self.device = device
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            device=device,
                            batch_first=batch_first,
                            dropout=dropout)

        self.reset_hidden_cell(batch_size)
        self.reset_hidden_state(batch_size)

    def reset_hidden_cell(self, batch_size):
        # You can pass the batch_size as an argument to allow for variable sized batches
        self.hidden_cell = torch.zeros((self.lstm.num_layers * self.num_directions), batch_size,
                                       self.hidden_size, device=self.device)

    def reset_hidden_state(self, batch_size):
        # You can pass the batch_size as an argument to allow for variable sized batches
        self.hidden_state = torch.zeros((self.lstm.num_layers * self.num_directions), batch_size,
                                        self.hidden_size, device=self.device)

    def forward(self, input_seq):
        """The current hidden states and cells are passed in the forward pass."""
        lstm_out, (self.hidden_cell, self.hidden_state) = self.lstm(input_seq,
                                                                    (self.hidden_cell, self.hidden_state))

        return lstm_out, (self.hidden_cell, self.hidden_state)


class LSTMStatefulClassificationOriginalSequence(nn.Module):
    """Create a stateful LSTM model for binary classification where the current hidden states and values
      are passed on to the next batch in the forward pass. Hidden states and cells are resettable to zero
      by a method.

      Padded sequences will be masked for the lstm using the function pack_padded_sequence.

      Bias of forget gate is initialized at 1 for learning long-term dependencies:

      'A Simple Way to Initialize Recurrent Networks of Rectified Linear Units' paper for reference.

      Parameters
      ----------
      input_size : int
          Input dimension of LSTM model.
      hidden_size : int
          Hidden states dimensionality.
      output_size : int
          Number of labels you want to predict.
      batch_size : int
          Number of samples in a batch.
      device : str
          'cuda' or 'cpu'
      num_layers : int
          Number of layers in the LSTM model. num_layers>1 makes it a stacked LSTM model.
      batch_first : Boolean
          True or False. Default True.
      bidirectional : Boolean
          True or False.
      dropout : float
          Dropout after each activation layer in the LSTM model.
    """

    def __init__(self, input_size, hidden_size, output_size, batch_size, device,
                 num_layers, batch_first, bidirectional, dropout):
        super().__init__()
        self.lstm = LSTMStateful(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 bidirectional=bidirectional,
                                 batch_size=batch_size, device=device,
                                 batch_first=batch_first,
                                 dropout=dropout)

        # Fully connected layer
        self.linear = nn.Linear(hidden_size * self.lstm.num_directions, output_size)

        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        Taken from: https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        To set forget-gate bias to 1: https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'linear' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x, x_lengths):
        # Clamp everything to minimum length of 1, but keep the original variable (x_lengths) to mask the
        # output later
        # Source: https://github.com/pytorch/pytorch/issues/4582
        # It is necessary to do so because the pack_padded_sequence function cannot handle sequences of
        # length 0 (therefore we give it a length of 1)
        x_lengths_clamped = x_lengths.clamp(min=1, max=None)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        packed_x = pack_padded_sequence(x, x_lengths_clamped.cpu().numpy(), batch_first=True)

        # LSTM returns hidden state of all timesteps (which is the variable output), as well as the
        # hidden state at last timestep (which is the variable hidden_state).
        # However, the variable output will also contain the padded hidden states (so the 0s).
        # The variable hidden_state, however, contains all non-zero output from LSTM.
        # Therefore, we need to use the variable hidden_state for our Linear layer (as we do not
        # want the padded values.)
        # Source: https://discuss.pytorch.org/t/how-to-use-pack-padded-sequence-correctly-how-to-compute-the-loss/38284
        # Hidden_state is of shape [num_layers, batch_size, hidden_dim]

        output, (h_t, c_t) = self.lstm(packed_x)

        last_layer_hidden_state = h_t.view(self.lstm.lstm.num_layers, self.lstm.num_directions,
                                           x.size(0), self.lstm.hidden_size)[-1]

        # MASKING HERE
        # mask everything that had seq_length as 0 in input as 0
        # last hidden_state is of shape [batch_size, hidden_dim]
        last_layer_hidden_state.masked_fill_((x_lengths == 0).view(-1, 1), 0)

        # Handle directions
        final_hidden_state = None
        if self.lstm.num_directions == 1:
            final_hidden_state = last_layer_hidden_state.squeeze()
        elif self.lstm.num_directions == 2:
            # h_1 and h_2 represent the last hidden states for the forward and backward pass in case
            # of a bidirectional LSTM.
            h_1, h_2 = last_layer_hidden_state[0], last_layer_hidden_state[1]
            final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states

        output = self.linear(final_hidden_state)

        return output


class LSTMStatefulClassificationFeatureSequence(nn.Module):
    """Create a stateful LSTM model for binary classification where the current hidden states and values
      are passed on to the next batch in the forward pass. Hidden states and cells are resettable to zero
      by a method.

      Parameters
      ----------
      input_size : int
          Input dimension of LSTM model.
      hidden_size : int
          Hidden states dimensionality.
      output_size : int
          Number of labels you want to predict.
      batch_size : int
          Number of samples in a batch.
      device : str
          'cuda' or 'cpu'
      num_layers : int
          Number of layers in the LSTM model. num_layers>1 makes it a stacked LSTM model.
      batch_first : Boolean
          True or False. Default True.
      bidirectional : Boolean
          True or False.
      dropout : float
          Dropout after each activation layer in the LSTM model.
    """

    def __init__(self, input_size, hidden_size, output_size, batch_size, device,
                 num_layers, batch_first, bidirectional, dropout):
        super().__init__()
        self.lstm = LSTMStateful(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 bidirectional=bidirectional,
                                 batch_size=batch_size, device=device,
                                 batch_first=batch_first,
                                 dropout=dropout)

        # Fully connected layer
        self.linear = nn.Linear(hidden_size * self.lstm.num_directions, output_size)

        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        Taken from: https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        To set forget-gate bias to 1: https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'linear' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        # LSTM returns hidden state of all timesteps (which is the variable output), as well as the
        # hidden state at last timestep (which is the variable hidden_state).
        # However, the variable output will also contain the padded hidden states (so the 0s).
        # The variable hidden_state, however, contains all non-zero output from LSTM.
        # Therefore, we need to use the variable hidden_state for our Linear layer (as we do not
        # want the padded values.)
        # Source: https://discuss.pytorch.org/t/how-to-use-pack-padded-sequence-correctly-how-to-compute-the-loss/38284
        # Hidden_state is of shape [num_layers * num_directions, batch_size, hidden_dim]

        output, (h_t, c_t) = self.lstm(x)

        last_layer_hidden_state = h_t.view(self.lstm.lstm.num_layers, self.lstm.num_directions,
                                           x.size(0), self.lstm.hidden_size)[-1]

        # Handle directions
        final_hidden_state = None
        if self.lstm.num_directions == 1:
            final_hidden_state = last_layer_hidden_state.squeeze()
        elif self.lstm.num_directions == 2:
            # h_1 and h_2 represent the last hidden states for the forward and backward pass in case
            # of a bidirectional LSTM.
            h_1, h_2 = last_layer_hidden_state[0], last_layer_hidden_state[1]
            final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states

        output = self.linear(final_hidden_state)

        return output


class TCNCombinedModelCopies(nn.Module):
    """ Model that combines both sequential and static data. Sequential data is processed according to a
    TCN model and hereafter the sequential and static data are concatenated and pushed through a linear layer.

    Parameters
    ----------
    input_size : int
        Input dimension of the time series for the TCN model. (Number of features you want to include).
    output_size : int
        Number of classes you want to predict.
    num_channels : List[int]
    stride : int
    kernel_size : int
    dropout : float
    input_dim_static : int
        Input dimension of the static data. (Number of features you want to include).
    hidden_dim_combined : int
        Hidden number of units for the combination of seq + static data.
    model_optional : nn.Sequential(*layers)
    """
    def __init__(self, input_size, output_size, num_channels, stride, kernel_size, dropout,
                 hidden_dim_combined, model_optional):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, stride, kernel_size=kernel_size, dropout=dropout)
        self.output_size = output_size
        self.hidden_dim_combined = hidden_dim_combined

        # Fully connected layer for the combined data
        self.fc_combined = nn.Linear(num_channels[-1], self.hidden_dim_combined)

        self.model_optional = model_optional

        for name, layer in self.model_optional.named_modules():
            if not isinstance(layer, nn.Linear):
                self.fc_output = nn.Linear(self.hidden_dim_combined, self.output_size)

            if isinstance(layer, nn.Linear):
                self.fc_output = nn.Linear(layer.weight.size(dim=0), self.output_size)
                break

        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        Taken from: https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        To set forget-gate bias to 1: https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        """
        for name, p in self.named_parameters():
            try:
                if 'fc' or 'model_optional' in name:
                    if 'weight' in name:
                        nn.init.xavier_uniform_(p.data)
                    elif 'bias' in name:
                        p.data.fill_(0)

            # This ValueError is for the case when model_optional is empty (i.e., nn.Sequential())
            except ValueError as e:
                if str(e) == "ValueError: Fan in and fan out can not be computed for tensor with fewer than 2 dimensions":
                    break

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in), where N is the batch size, C_in is the number of features or
        channels, L_in is the seq_len"""
        if inputs.isnan().any():
            print(f'Inputs forward pass: {inputs}')
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        if y1.isnan().any():
            print(f'Y1 forward pass: {y1}')

        # Through the linear layer
        x_combined = self.fc_combined(y1[:, :, -1])  # Return output of last time step

        # Optional model part
        x_combined = self.model_optional(x_combined)

        # Final linear layer
        output_combined = self.fc_output(x_combined)

        return output_combined


class TCNCombinedModel(nn.Module):
    """ Model that combines both sequential and static data. Sequential data is processed according to a
    TCN model and hereafter the sequential and static data are concatenated and pushed through a linear layer.

    Parameters
    ----------
    input_size : int
        Input dimension of the time series for the TCN model. (Number of features you want to include).
    output_size : int
        Number of classes you want to predict.
    num_channels : List[int]
    stride : int
    kernel_size : int
    dropout : float
    input_dim_static : int
        Input dimension of the static data. (Number of features you want to include).
    hidden_dim_combined : int
        Hidden number of units for the combination of seq + static data.
    model_optional : nn.Sequential(*layers)
    """
    def __init__(self, input_size, output_size, num_channels, stride, kernel_size, dropout,
                 input_dim_static, hidden_dim_combined, model_optional):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, stride, kernel_size=kernel_size, dropout=dropout)
        self.output_size = output_size
        self.hidden_dim_combined = hidden_dim_combined

        # Fully connected layer for the combined data
        self.fc_combined = nn.Linear(num_channels[-1] + input_dim_static, self.hidden_dim_combined)

        self.model_optional = model_optional

        for name, layer in self.model_optional.named_modules():
            if not isinstance(layer, nn.Linear):
                self.fc_output = nn.Linear(self.hidden_dim_combined, self.output_size)

            if isinstance(layer, nn.Linear):
                self.fc_output = nn.Linear(layer.weight.size(dim=0), self.output_size)
                break

        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        Taken from: https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        To set forget-gate bias to 1: https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        """
        for name, p in self.named_parameters():
            try:
                if 'fc' or 'model_optional' in name:
                    if 'weight' in name:
                        nn.init.xavier_uniform_(p.data)
                    elif 'bias' in name:
                        p.data.fill_(0)

            # This ValueError is for the case when model_optional is empty (i.e., nn.Sequential())
            except ValueError as e:
                if str(e) == "ValueError: Fan in and fan out can not be computed for tensor with fewer than 2 dimensions":
                    break

    def forward(self, inputs, x_static):
        """Inputs have to have dimension (N, C_in, L_in), where N is the batch size, C_in is the number of features or
        channels, L_in is the seq_len"""
        if inputs.isnan().any():
            print(f'Inputs forward pass: {inputs}')
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        if y1.isnan().any():
            print(f'Y1 forward pass: {y1}')

        # Here we add the static data to the processed (by the TCN) time series data
        x_combined = torch.concat([y1[:, :, -1], x_static], dim=1)

        # Through the linear layer
        x_combined = self.fc_combined(x_combined)

        # Optional model part
        x_combined = self.model_optional(x_combined)

        # Final linear layer
        output_combined = self.fc_output(x_combined)

        return output_combined


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn_1 = nn.BatchNorm1d(n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn_2 = nn.BatchNorm1d(n_outputs)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn_1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.bn_2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # Source on how to initialize: https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        conv1 = self.conv1(x)
        if conv1.isnan().any():
            print(f'Conv1 forward pass: {conv1}')

        chomp1 = self.chomp1(conv1)
        if chomp1.isnan().any():
            print(f'Chomp1 forward pass: {chomp1}')

        relu1 = self.relu1(chomp1)
        if relu1.isnan().any():
            print(f'Relu1 forward pass: {relu1}')

        dropout1 = self.dropout1(relu1)
        if dropout1.isnan().any():
            print(f'Dropout1 forward pass: {dropout1}')

        conv2 = self.conv2(dropout1)
        if conv2.isnan().any():
            print(f'Conv2 forward pass: {conv2}')
            print(list(self.conv2.named_parameters()))

        chomp2 = self.chomp2(conv2)
        if chomp2.isnan().any():
            print(f'Chomp2 forward pass: {chomp2}')

        relu2 = self.relu2(chomp2)
        if relu2.isnan().any():
            print(f'relu2 forward pass: {relu2}')

        dropout2 = self.dropout2(relu2)
        if dropout2.isnan().any():
            print(f'dropout2 forward pass: {dropout2}')

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


def get_num_levels_based_on_receptive_field(kernel_size: int, receptive_field: int, dilation_exponential_base: int):
    # Calculate the num_levels (num residual blocks) needed to match at least 90% of the desired receptive_field
    # Calculation is taken from:
    # https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-3-7f6633fcc7c7
    levels = math.ceil((math.log((receptive_field - receptive_field * 0.1) / (1 + 2 * (kernel_size - 1))) / math.log(
        dilation_exponential_base)) + 1)

    return levels


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, stride, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, stride, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, stride, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in), where N is the batch size, C_in is the number of features or
        channels, L_in is the seq_len"""
        if inputs.isnan().any():
            print(f'Inputs forward pass: {inputs}')
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        if y1.isnan().any():
            print(f'Y1 forward pass: {y1}')
        o = self.linear(y1[:, :, -1])  # Return output of last time step

        # As we use BCEwithlogitsloss (which has an internal sigmoid function)
        # we return the output of the last linear layer
        return o
