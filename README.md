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
