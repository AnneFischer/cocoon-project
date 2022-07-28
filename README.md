# cocoon-project
Repo for preterm birth prediction using machine learning.

Download this repo by running: ```git clone https://github.com/AnneFischer/cocoon-project```.

## Download TPEHG database from PhysioNet
Data can be found here: [PhysioNet](https://physionet.org/content/tpehgdb/1.0.1/)

The structure of the directory should be: ```{your_data_path}/tpehgdb/tpehg<ID>.{hea,dat}```.

You should add your_data_path to the file_paths.json file, which can be found under 

```python
references/settings/file_paths.json
```

## Run the following command to install the required dependencies: 

```bash
pip install -e .
```

or

```bash
pip3 install -e .
```
### Add the correct paths 
In the ```file_paths.json``` file you should specify your settings_path and output_path. All results will be saved in your output_path.

## Step 1: Preprocess data

We preprocess the EHG data from the TPEHG DB by using a fourth order Butterworth filter with a bandwith of 0.34 - 1 Hz. To obtain the filtered data run:

```python
python ./src_pre_term_database/preprocess_data.py
```

Running this file may take a couple of minutes and the resulting file will be saved under the name `df_signals_filt.csv` in the data_path folder you specified in the `file_paths.json` file. The first and last 3600 seconds of each recording will be removed because of transient effects of the filtering.

## Step 2: Evaluation

The final trained models are provided in the `trained_models` folder. The final models are evaluated on the test set and performance is given in terms of AUC and AP. The optimal hyperparameters are provided in the `best_params.json` file and the optional_model part (when EHG data and clinical data are combined) has to be put in the optional_model_dict and bidirectional_lstm_dict, which are placed at the top of the `evaluation.py` file. The name of your final model has to be specified in the `final_models.json` file. 

Usage:

```
  --model {tcn,lstm}    Select what model to use: 'lstm' or 'tcn'
  
  --feature_name {sample_entropy,peak_frequency,median_frequency}
                        Select what feature to use for data reduction: 'sample_entropy', 'peak_frequency' or 'median_frequency'
                        
  --add_static_data     Add static clinical data to the model. Use either the --add_static_data or the--no_static_data flag
  
  --no_static_data      Use only the EHG data for modeling. Use either the --add_static_data or the--no_static_data flag
  
  --use_copies_for_static_data
                        The static data is now treated as a time series, were each (static) value of each variable is copied
                        along the time steps of the EHG time series data.Meaning, if there are 10 time steps in the seq data,
                        then the static data is also copied for 10 time steps.
                        
  --no_copies_for_static_data
                        The static data is now treated as single values that will be concatenated separately to the time series
                        data after the time series data has been processed. Use either the --use_copies_for_static_data or the
                        --no_copies_for_static_data flag.
                        
```

Example of how to evaluate the final model for TCN with peak frequency as method of data reduction, no static data added in the command line:

```python
python ./src_pre_term_database/evaluation.py --model 'tcn' --feature_name 'peak_frequency' --no_static_data
```


