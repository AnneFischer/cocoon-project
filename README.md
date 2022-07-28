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
