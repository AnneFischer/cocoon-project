from src_pre_term_database.load_dataset import build_signal_dataframe
from src_pre_term_database.data_processing_and_feature_engineering import create_filtered_channels, \
    remove_first_n_samples_of_signals, remove_last_n_samples_of_signals
from src_pre_term_database.utils import read_settings
import os

settings_path = os.path.abspath("references/settings")

file_paths = read_settings(settings_path, 'file_paths')
data_path = file_paths['data_path']


if __name__ == "__main__":
    df_signals = build_signal_dataframe(data_path, settings_path)
    # Filter the data using a fourth order Butterworth filter using the following bandwidths:
    # 0.34-1 Hz
    # 0.08-4 Hz
    # 0.3-3 Hz
    # 0.3-4 Hz
    # Only the 0.34-1 Hz bandpass filter will be used
    df_signals_new = create_filtered_channels(df_signals, ['channel_1', 'channel_2', 'channel_3'],
                                              [[0.34, 1], [0.08, 4], [0.3, 3], [0.3, 4]], fs=20, order=4)
    # Remove first and last 3 minutes of recording because of transient effects of filtering
    df_signals_new = remove_first_n_samples_of_signals(df_signals_new, n=3600)
    df_signals_new = remove_last_n_samples_of_signals(df_signals_new, n=3600)

    df_signals_new.to_csv(f'{data_path}/df_signals_filt.csv', sep=';')
