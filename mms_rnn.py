import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pytplot
import pyspedas


def mms_rnn_data(probe_id: int, datadir: str) -> None:
    """Generates csv file of data suitable for RNN.

    Args:
        probe_id (int): MMS spacecraft ID
        datadir (str): data load/save directory

    Returns:
        None
    """

    pyspedas.tinterpol('mms' + str(probe_id) + '_B_norm',
                       'mms' + str(probe_id) + '_ve_norm',
                       method='linear',
                       newname='mms' + str(probe_id) + '_B_norm')
    pyspedas.tinterpol('mms' + str(probe_id) + '_E_norm',
                       'mms' + str(probe_id) + '_ve_norm',
                       method='linear',
                       newname='mms' + str(probe_id) + '_E_norm')
    pyspedas.tinterpol('mms' + str(probe_id) + '_Ni_norm',
                       'mms' + str(probe_id) + '_ve_norm',
                       method='linear',
                       newname='mms' + str(probe_id) + '_Ni_norm')
    pyspedas.tinterpol('mms' + str(probe_id) + '_Ne_norm',
                       'mms' + str(probe_id) + '_ve_norm',
                       method='linear',
                       newname='mms' + str(probe_id) + '_Ne_norm')
    pyspedas.tinterpol('mms' + str(probe_id) + '_vi_norm',
                       'mms' + str(probe_id) + '_ve_norm',
                       method='linear',
                       newname='mms' + str(probe_id) + '_vi_norm')

    t, B = pytplot.get_data('mms' + str(probe_id) + '_B_norm')
    _, E = pytplot.get_data('mms' + str(probe_id) + '_E_norm')
    _, ni = pytplot.get_data('mms' + str(probe_id) + '_Ni_norm')
    _, ne = pytplot.get_data('mms' + str(probe_id) + '_Ne_norm')
    _, vi = pytplot.get_data('mms' + str(probe_id) + '_vi_norm')
    _, ve = pytplot.get_data('mms' + str(probe_id) + '_ve_norm')

    ve = ve/np.sqrt(1836)

    nn_data = pd.DataFrame(index=pd.to_datetime(t, unit='s'),
                    data = {'Bx': B[:, 0], 'By': -B[:, 2], 'Bz': B[:, 1],
                            'Ex': E[:, 0], 'Ey': -E[:, 2], 'Ez': E[:, 1],
                            'ni': ni, 'ne': ne,
                            'vix': vi[:, 0], 'viy': -vi[:, 2], 'viz': vi[:, 1],
                            'vex': ve[:, 0], 'vey': -ve[:, 2], 'vez': ve[:, 1]})
    
    nn_data.to_csv(datadir + '/mms' + str(probe_id) + '_nndata.csv')

    return None

def mms_rnn_label(probe_id: int, datadir: str) -> None:
    """Runs prepared, scaled data for an event through optimal RNN.

    Args:
        probe_id (int): MMS spacecraft ID
        datadir (str): data load/save directory

    Returns:
        None
    """

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(datadir + '/mms' + str(probe_id) + '_nndata.csv',
                     header=0, index_col=0)

    data = df[['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez', 'ni', 'ne', 'vix', 'viy',
               'viz', 'vex', 'vey', 'vez']].values
    time_series_list = data

    class RNNModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size,
                     num_layers, dropout_prob):
            super(RNNModel, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout_prob)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = self.fc(out)
            return out

    sc_data = torch.tensor(time_series_list, dtype=torch.float32).to(device)

    # Initialize the model with the best hyperparameters
    best_input_size = sc_data[0].shape[0]
    best_hidden_size = 42
    best_output_size = 6
    best_dropout_prob = 3.538743923225951e-06
    best_num_layers = 3

    best_model = RNNModel(best_input_size, best_hidden_size, best_output_size,
                          best_num_layers, best_dropout_prob).to(device)

    best_model.load_state_dict(torch.load('best_model'))

    # Set the model to evaluation mode
    best_model.eval()

    # Create a DataLoader for sc data
    sc_dataset = TensorDataset(sc_data)
    sc_dataloader = DataLoader(sc_dataset, batch_size=32, shuffle=False)

    # Lists to store predictions and true labels
    all_predictions = []
    all_labels = []

    prob_preds = []
    probabilities = []

    with torch.no_grad():
        for inputs in sc_dataloader:
            # Forward pass
            outputs = best_model(inputs[0])
            outputs = outputs.view(-1, best_output_size)

            probs = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)

            predicted = torch.argmax(outputs, dim=1)

            # Convert predictions and labels to CPU and store them
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(inputs[0].cpu().numpy())

            prob_preds.extend(predicted_class.cpu().numpy())
            probabilities.extend(confidence.cpu().numpy())

    results_df = pd.DataFrame({'Predicted': all_predictions})
    results_df.to_csv(datadir + '/mms' + str(probe_id) + '_labels.csv',
                      index=False)
    
    probs_df = pd.DataFrame({'Predicted': prob_preds, 'Confidence': probabilities})
    probs_df.to_csv(datadir + '/mms' + str(probe_id) + '_probs.csv',
                    index=False)

    return None