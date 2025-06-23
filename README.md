# MMS Reconnection Region Classification with RNN

![image](https://github.com/user-attachments/assets/393c166a-f0ec-487a-8af1-74667eb6d6f6)

This repository contains a pipeline to process MMS burst data, normalise and scale it for comparison across events, and apply a trained recurrent neural network (RNN) model to classify different regions in magnetotail reconnection.

Event list is sourced from Supporting Information of Rogers+ 2023, https://doi.org/10.1029/2022JA030577

## 📁 Repository Structure

├── rnn_run.py # Main entry point; runs the full pipeline for selected events \
├── mms_data_load.py # Loads and preprocesses MMS burst data \
├── mms_event_scale.py # Scales data using normalisation constants (B0, n0) \
├── mms_rnn.py # Formats the data for RNN input and applies the trained RNN \
├── best_model # Trained PyTorch RNN model used for inference \
├── data/ # Directory where intermediate and output files are saved \

## 🔧 Requirements
This project requires Python (≥3.8) and a set of scientific and space physics packages. You can set it up using either a Conda environment or a standard Python virtual environment.

### 🟦 Option 1: Using Conda (recommended)

1. Create and activate a new environment:
```bash
conda create -n mms-rnn python=3.10
conda activate mms-rnn
```

2. Install dependencies:
```bash
conda install numpy pandas matplotlib pip
pip install torch pytplot pyspedas
```

### 🟨 Option 2: Using a standard Python environment

1. Create and activate a virtual environment:
```bash
python -m venv mms-rnn-env
source mms-rnn-env/bin/activate     # On macOS/Linux
mms-rnn-env\Scripts\activate.bat    # On Windows
```

2. Upgrade pip and install dependencies:
```bash
pip install --upgrade pip
pip install numpy pandas matplotlib torch pytplot pyspedas
```

## ▶️ How to Run
Run the pipeline with:

```bash
python rnn_run.py
```

This script:

Loads MMS burst and field data for a set of predefined events using mms_data_load.py

Computes key parameters (currents, J·E, etc.) and saves them

Normalises and scales the data using event-specific B₀ and n₀ (mms_event_scale.py)

Formats the scaled data into CSV for use with the RNN (mms_rnn_data)

Runs a pretrained RNN on the data and outputs predicted region classifications (mms_rnn_label)

All intermediate and output files (CSV and tplot save files) are saved to the data/ directory, organised by event name and spacecraft ID.

## 📦 Output
For each event, the following files are saved in data/{EVENT}_mms{N}/:

mms{N}_nndata.csv: Normalized time series data formatted for the RNN

mms{N}_labels.csv: Region label predicted by the RNN at each timestep

mms{N}_probs.csv: Predicted class and confidence values from the softmax layer

mms{N}_vars.tplot: Raw and processed data

mms{N}_vars_norm.tplot: Scaled data for cross-event comparison

## 📘 Notes
The pretrained RNN (best_model) assumes a 14-feature input corresponding to field, density, and velocity variables.

The normalization is event-specific to allow comparison across MMS tail events with varying plasma conditions.

By default, only a subset of events (IDs given by nums in rnn_run.py) is processed to reduce runtime.

## 📄 Author
Cara Waters, 2025 \
Imperial College London \
For questions, please contact: cara.waters18@imperial.ac.uk
