import pandas as pd
import numpy as np
from typing import List, Union, Tuple
from pathlib import Path
import constants as c
import os
from re import match
import wfdb
from tqdm.auto import tqdm
from utils import rename_columns_from_mappings, read_settings, convert_columns_to_numeric


settings_path = '/Users/AFischer/PycharmProjects/cocoon-project/references/settings'

file_paths = read_settings(settings_path, 'file_paths')

SIGNAL_COLUMN_NAMES = ['1', '1_DOCFILT-4-0.08-4', '1_DOCFILT-4-0.3-3', '1_DOCFILT-4-0.3-4',
                      '2', '2_DOCFILT-4-0.08-4', '2_DOCFILT-4-0.3-3', '2_DOCFILT-4-0.3-4',
                      '3', '3_DOCFILT-4-0.08-4', '3_DOCFILT-4-0.3-3', '3_DOCFILT-4-0.3-4']

VARIABLE_CONSTANTS_LIST = ['RecID', 'Age', 'Parity', 'Abortions', 'Weight',
                           'Hypertension', 'Diabetes', 'Placental_position', 'Bleeding_first_trimester',
                           'Bleeding_second_trimester', 'Funneling', 'Smoker']


def create_file_list_database(path_to_signals: Union[str, Path]) -> List:
    """Create a list of all the record ids from the term-preterm database
    from PhysioNet.

    Parameters
    ----------
    path_to_signals : str
        Path to folder with EHG signal patient files.

    Returns
    -------
    filelist : List
        List containing all the record ids from the term preterm database.
    """
    # The data comes from PhysioNet and in order to read in the data
    # we need to use the WFDB python package.
    # The name of the WFDB record (record id) that has to be read must not contain any file extensions,
    # therefore we will strip of the .dat and .hea extension.
    filelist = os.listdir(path_to_signals)  # this list contains all wfdb records (including extension)
    filelist = [i.replace('.dat', "").replace('.hea', '') for i in filelist]  # strip of .dat and .hea

    # The final list will contain all record ids from the database (without the file extensions).
    filelist = list(dict.fromkeys(filelist))

    return filelist


def split_rec_id(df):
    """Split the rec_id column such that it only contains the integer
    and not the prefix 'tpehg'. Example: Rec ids that have the form
    tpehg1007, will be split such that the returned rec id will be just
    1007.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains the column 'rec_id'.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with the column rec_id that only contains the integer.
    """
    # New data frame where the rec_id is split into 'tpehg' and the corresponding
    # integer
    new_df = df[c.REC_ID_NAME].str.split("tpehg", n=1, expand =True)

    # We need to make sure that the replacement of the rec_ids is correct. If the length
    # of the new rec_ids does not match the length of the old rec_ids we know something
    # went wrong.
    assert (len(df[c.REC_ID_NAME]) == len(new_df[1])), "The rec_ids are not properly split!"

    # The second column of the new dataframe contains the integer, and we replace
    # the old 'rec_id' of df with the new 'rec_id' that only consists of the integer.
    df[c.REC_ID_NAME] = new_df[1]

    return df


def build_signal_dataframe(path_to_data: str, path_to_settings: str) -> pd.DataFrame:
    """Loop over all patient EHG signal files and combine all EHG recordings into
    one dataframe and save it.

    Parameters
    ----------
    path_to_data : str
        Path to folder with the term-preterm database files.
    path_to_settings : str
        Path to settings folder where all the settings files are saved.

    Returns
    -------
    df_final_signals : pd.DataFrame
        Dataframe that contains all EHG signals from all record ids.
    """
    # Create a Path object in order to convert to the correct OS path.
    data_path = Path(f'{path_to_data}')

    path_to_signals = data_path / "tpehgdb"
    # This filelist contains all the record ids from the database.
    filelist = create_file_list_database(path_to_signals)

    # In this dataframe we will store the EHG signals from all WFDB records.
    df_final_signals = pd.DataFrame(columns=[c.REC_ID_NAME] + SIGNAL_COLUMN_NAMES)

    for id_file in tqdm(filelist):
        # In the signals variable, the EHG data from all channels are saved and in
        # the fields variable all the record descriptors (e.g., age, gestation, etc.)
        # of the patients is stored.
        signals, fields = wfdb.rdsamp(f'{path_to_signals}/{id_file}')
        df_signals = pd.DataFrame(signals, columns=SIGNAL_COLUMN_NAMES)

        # As we also want to know which record id belongs to which EHG data, we also save the id_file
        # in df_signals.
        df_signals = pd.concat([pd.Series([id_file] * len(signals)).to_frame(c.REC_ID_NAME), df_signals], axis=1)

        df_final_signals = pd.concat([df_final_signals, df_signals], ignore_index=True)

    column_names_mappings: dict = read_settings(path_to_settings, 'column_names_mappings')
    df_final_signals = rename_columns_from_mappings(df_final_signals, column_names_mappings['records'])

    df_final_signals = split_rec_id(df_final_signals)
    df_final_signals = convert_columns_to_numeric(df_final_signals, [c.REC_ID_NAME])

    return df_final_signals


def build_demographics_dataframe(path_to_data: str,
                                 path_to_settings: str) -> pd.DataFrame:
    """Loop over all patient EHG signal files and combine all demograhpic data of the
    records in one dataframe and save it.

    Parameters
    ----------
    path_to_data : str
        Path to folder with the term-preterm database files.
    path_to_settings : str
        Path to settings folder where all the settings files (i.e., the column names mappings, etc.)
        are saved.

    Returns
    -------
    df_final_demographics : pd.DataFrame
        Dataframe that contains all demographic data (e.g., age, gestation, etc.)
        from all record ids.
    """
    # Create a Path object in order to convert to the correct OS path.
    data_path = Path(f'{path_to_data}')
    path_to_signals = data_path / "tpehgdb"

    # This filelist contains all the record ids from the database.
    filelist = create_file_list_database(path_to_signals)

    df_final_demographics = pd.DataFrame(columns=VARIABLE_CONSTANTS_LIST)
    for id_file in tqdm(filelist):
        # In the signals variable, the EHG data from all channels are saved and in
        # the fields variable all the static information (e.g., age, gestation, etc.)
        # of the patients is stored.
        signals, fields = wfdb.rdsamp(f'{path_to_signals}/{id_file}')

        row_to_add = []
        for key, value in fields.items():
            # In the 'comments' key all the demographic data such as the age, gestation, etc.
            # are stored.
            if str(key).startswith('comments'):
                # The variables of interest are stored in VARIABLE_CONSTANT_LIST and
                # we want to store all values for those variables in a dataframe.
                for variable in VARIABLE_CONSTANTS_LIST:
                    # variable_list will have the form of [variable value], and we want to split this
                    # into the form [variable, value] (this is done in variable_list_split).
                    variable_list = list(filter(lambda v: match(variable, v), value))
                    variable_list_split = [word for line in variable_list for word in line.split()]

                    value_of_variable = f'{variable_list_split[1]}'  # The second element of the list contains the value
                    row_to_add.append(value_of_variable)

                # After we've obtained all values from the variables of interest, we put
                # them in a pd.Series format and append those to the final dataframe.
                row_to_add_series = pd.Series(row_to_add, index=df_final_demographics.columns)

                df_final_demographics = df_final_demographics.append(row_to_add_series, ignore_index=True)

        column_names_mappings: dict = read_settings(path_to_settings, 'column_names_mappings')
        df_final_demographics = rename_columns_from_mappings(df_final_demographics,
                                                             column_names_mappings['records'])
        df_final_demographics = convert_columns_to_numeric(df_final_demographics,
                                                           [c.REC_ID_NAME, c.PARITY_NAME, c.AGE_NAME,
                                                            c.ABORTIONS_NAME, c.WEIGHT_NAME], errors='coerce')

    return df_final_demographics


def build_clinical_information_dataframe(path_to_data: str,
                                         path_to_settings: str) -> pd.DataFrame:
    """Create a dataframe containing the following additional clinical information:

    Record - the name of the record;
    Gestation - pregnancy duration (in weeks);
    Rec. time - gestation duration at the time of recording (in weeks);
    Group - record group according to gestation duration at the time of recording (<26 weeks, >=26 weeks) and
    pregnancy duration (PRE: pre-term, TERM: term);
    Premature - true (t), if delivery was premature (before 37 weeks of gestation); false (f), otherwise;
    Early - true (t), if the record was obtained before the 26th week of gestation; false (f), otherwise.

    Parameters
    ----------
    path_to_data : str
        Path to folder with the term-preterm database files.
    path_to_settings : str
        Path to settings folder where all the settings files (i.e., the column names mappings, etc.)
        are saved.

    Returns
    -------
    df_clinical : pd.DataFrame
        Dataframe that contains all clinical data (e.g., pregnancy duration, etc.)
        from all record ids.
    """
    # Create a Path object in order to convert to the correct OS path.
    data_path = Path(f'{path_to_data}')
    path_to_clinical_information = data_path / "tpehgdb.smr"

    df_clinical = pd.read_csv(f'{path_to_clinical_information}', sep='|', skiprows=[1])

    column_names_mappings: dict = read_settings(path_to_settings, 'column_names_mappings')
    df_clinical = rename_columns_from_mappings(df_clinical, column_names_mappings['records'])
    df_clinical = split_rec_id(df_clinical)
    df_clinical = convert_columns_to_numeric(df_clinical, [c.REC_ID_NAME])

    return df_clinical


def split_term_preterm_rec_ids(df: pd.DataFrame,
                               preterm_threshold: np.float = 37.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into two new dataframes: one dataframe containing the term rec_ids and the other
    dataframe containing the preterm rec_ids. The split is based on the preterm_threshold, which is the
    threshold for determining when a patient has delivered preterm. The default value is 37.0 weeks.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing all the rec_ids. This dataframe MUST contain the column 'gestation',
        which is the gestation duration in weeks.
    preterm_threshold : np.float
        Threshold value of when a patient delivered preterm. Default value is 37.0 weeks.

    Returns
    -------
    df_preterm, df_term : Tuple[pd.DataFrame, pd.DataFrame]
        Dataframes that contain the preterm rec_ids and term rec_ids respectively.
    """
    df_preterm = df.query(f'{c.GESTATION_NAME}<{preterm_threshold}').reset_index(drop=True)
    df_term = df.query(f'{c.GESTATION_NAME}>={preterm_threshold}').reset_index(drop=True)

    return df_preterm, df_term
