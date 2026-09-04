# natural scenes data analysis functions 

import pickle
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
import pandas as pd


def load_movie_current_clamp_data(filename):
    # loads MATLAB movie current-clamp data
    # input
    # filename is the path to the MATLAB .mat file
    
    # output
    # traces_raw is the raw MATLAB traces object
    # groups_raw is an array of group labels for each recorded cell
    
    mat = loadmat(filename, squeeze_me=True, struct_as_record=False)
    traces_raw = mat["traces"]
    groups_raw = np.asarray(mat["RecGroup"]).squeeze()
    
    return traces_raw, groups_raw


def unpack_movie_current_clamp(traces_raw, groups_raw):
    # converts MATLAB cell-like movie current-clamp traces into trial-wise traces and metadata
    # input
    # traces_raw is the raw MATLAB traces object containing one entry per recorded cell
    # groups_raw is an array of group labels for each recorded cell
    
    # output
    # traces is an array of shape n_trials x n_time containing one trace per trial
    # meta is a DataFrame containing cell_id, trial_id, and group for each trace
    
    all_traces = []
    rows = []

    for cell_id, raw in enumerate(traces_raw):
        arr = np.asarray(raw)

        if arr.size == 0:
            continue

        arr = np.squeeze(arr)

        if arr.ndim == 1:
            all_traces.append(arr)
            rows.append({"cell_id": cell_id, "trial_id": 0, "group": int(groups_raw[cell_id])})

        elif arr.ndim == 2:
            n_time, n_trials = arr.shape

            for trial_id in range(n_trials):
                all_traces.append(arr[:, trial_id])
                rows.append({"cell_id": cell_id, "trial_id": trial_id, "group": int(groups_raw[cell_id])})

        else:
            raise ValueError(f"Unexpected shape at cell {cell_id}: {arr.shape}")

    traces = np.vstack(all_traces)
    meta = pd.DataFrame(rows)

    return traces, meta


def make_time_and_epoch_indices(n_time, duration_s=42, pre_grey_s=(0, 4), movie_s=(4, 39), post_grey_s=(39, 42)):
    # creates the time vector and Boolean indices for pre-movie, movie, and post-movie epochs
    # input
    # n_time is the number of sampled time points in each trace
    # duration_s is the total recording duration in seconds
    # pre_grey_s is a tuple with the start and end time of the pre-movie grey screen in seconds
    # movie_s is a tuple with the start and end time of the movie stimulus in seconds
    # post_grey_s is a tuple with the start and end time of the post-movie grey screen in seconds
    
    # output
    # time is an array of length n_time containing time values in seconds
    # fs_eff is the effective sampling frequency in Hz
    # pre_grey_idx is a Boolean array selecting the pre-movie grey epoch
    # movie_idx is a Boolean array selecting the movie epoch
    # post_grey_idx is a Boolean array selecting the post-movie grey epoch
    
    fs_eff = n_time / duration_s
    time = np.arange(n_time) / fs_eff
    pre_grey_idx = (time >= pre_grey_s[0]) & (time < pre_grey_s[1])
    movie_idx = (time >= movie_s[0]) & (time < movie_s[1])
    post_grey_idx = (time >= post_grey_s[0]) & (time < post_grey_s[1])
    
    return time, fs_eff, pre_grey_idx, movie_idx, post_grey_idx


def split_traces_by_group(traces, meta, group_col="group", group_values=(1, 2)):
    # splits traces and metadata by experimental group
    # input
    # traces is an array of shape n_trials x n_time containing one trace per trial
    # meta is a DataFrame containing metadata for each trace
    # group_col is the column name containing group labels
    # group_values is a tuple of group values to extract
    
    # output
    # traces_by_group is a dictionary mapping group values to trace arrays
    # meta_by_group is a dictionary mapping group values to metadata DataFrames
    
    traces_by_group = {group: traces[meta[group_col].to_numpy() == group] for group in group_values}
    meta_by_group = {group: meta.loc[meta[group_col] == group].copy() for group in group_values}
    
    return traces_by_group, meta_by_group


def compute_voltage_epoch_metrics(traces, meta, pre_grey_idx, movie_idx, post_grey_idx):
    # computes voltage fluctuation metrics for pre-movie, movie, and post-movie epochs
    # input
    # traces is an array of shape n_trials x n_time containing one trace per trial
    # meta is a DataFrame containing metadata for each trace
    # pre_grey_idx is a Boolean array selecting the pre-movie grey epoch
    # movie_idx is a Boolean array selecting the movie epoch
    # post_grey_idx is a Boolean array selecting the post-movie grey epoch
    
    # output
    # df_Vm is a DataFrame containing voltage fluctuation metrics for each trial
    
    rows = []

    for i, trace in enumerate(traces):
        row = meta.iloc[i]
        movie_trace = trace[movie_idx]

        rows.append({
            "trace_index": i,
            "cell_id": row["cell_id"],
            "trial_id": row["trial_id"],
            "group": row["group"],
            "Vm_std_pre": np.nanstd(trace[pre_grey_idx], ddof=1),
            "Vm_std_movie": np.nanstd(movie_trace, ddof=1),
            "Vm_std_post": np.nanstd(trace[post_grey_idx], ddof=1),
            "Vm_var_movie": np.nanvar(movie_trace, ddof=1),
            "Vm_mean_movie": np.nanmean(movie_trace),
            "Vm_abs_mean_movie": np.nanmean(np.abs(movie_trace))})

    df_Vm = pd.DataFrame(rows)

    return df_Vm


def detect_spikes(trace, time, epoch_idx, prominence=42, distance_ms=10):
    # detects spikes in a selected epoch of one voltage trace
    # input
    # trace is an array of membrane voltages
    # time is an array of time values in seconds
    # epoch_idx is a Boolean array selecting the epoch used for spike detection
    # prominence is the minimum spike prominence used by scipy.signal.find_peaks
    # distance_ms is the minimum distance between detected spikes in ms
    
    # output
    # spike_times is an array of detected spike times in seconds
    # peaks is an array of peak indices relative to the selected epoch
    # props is a dictionary of peak properties returned by scipy.signal.find_peaks
    
    dt = np.nanmean(np.diff(time))
    fs = 1 / dt
    segment = trace[epoch_idx]
    t_segment = time[epoch_idx]
    distance_samples = max(1, int(round(distance_ms / 1000 * fs)))
    peaks, props = find_peaks(segment, prominence=prominence, distance=distance_samples)
    spike_times = t_segment[peaks]
    
    return spike_times, peaks, props


def compute_trial_metrics(traces, meta, time, movie_idx, prominence=42, distance_ms=10):
    # computes spike and voltage metrics for each trial during the movie epoch
    # input
    # traces is an array of shape n_trials x n_time containing one trace per trial
    # meta is a DataFrame containing metadata for each trace
    # time is an array of time values in seconds
    # movie_idx is a Boolean array selecting the movie epoch
    # prominence is the minimum spike prominence used for spike detection
    # distance_ms is the minimum distance between detected spikes in ms
    
    # output
    # df_trial is a DataFrame containing spike and voltage metrics for each trial
    
    rows = []
    dt = np.nanmean(np.diff(time))
    movie_duration = np.sum(movie_idx) * dt

    for i, trace in enumerate(traces):
        row = meta.iloc[i]
        spike_times, peaks, props = detect_spikes(trace, time, movie_idx, prominence=prominence, distance_ms=distance_ms)
        n_spikes = len(spike_times)
        firing_rate = n_spikes / movie_duration
        movie_trace = trace[movie_idx]

        if n_spikes >= 3:
            isi = np.diff(spike_times)
            isi_mean = np.mean(isi)
            isi_var = np.var(isi, ddof=1)
            isi_std = np.std(isi, ddof=1)
            cv_isi = isi_std / isi_mean
        else:
            isi_mean = np.nan
            isi_var = np.nan
            isi_std = np.nan
            cv_isi = np.nan

        rows.append({
            "trace_index": i,
            "cell_id": row["cell_id"],
            "trial_id": row["trial_id"],
            "group": row["group"],
            "n_spikes_movie": n_spikes,
            "firing_rate_movie": firing_rate,
            "ISI_mean_movie": isi_mean,
            "ISI_var_movie": isi_var,
            "ISI_std_movie": isi_std,
            "CV_ISI_movie": cv_isi,
            "Vm_mean_movie": np.nanmean(movie_trace),
            "Vm_var_movie": np.nanvar(movie_trace, ddof=1),
            "Vm_std_movie": np.nanstd(movie_trace, ddof=1)})

    df_trial = pd.DataFrame(rows)

    return df_trial


def safe_var(x):
    # computes sample variance while safely handling NaN values and small sample sizes
    # input
    # x is an array-like object containing numerical values
    
    # output
    # variance is the sample variance of x or NaN if fewer than two values are available
    
    x = pd.Series(x).dropna().to_numpy(dtype=float)

    if len(x) <= 1:
        return np.nan

    variance = np.var(x, ddof=1)

    return variance


def safe_sem(x):
    # computes the standard error of the mean while safely handling NaN values and small sample sizes
    # input
    # x is an array-like object containing numerical values
    
    # output
    # sem_value is the standard error of the mean or NaN if fewer than two values are available
    
    x = pd.Series(x).dropna().to_numpy(dtype=float)

    if len(x) <= 1:
        return np.nan

    sem_value = np.std(x, ddof=1) / np.sqrt(len(x))

    return sem_value


def aggregate_cell_metrics(df_trial):
    # aggregates trial-level movie metrics to the cell level
    # input
    # df_trial is a DataFrame containing spike and voltage metrics for each trial
    
    # output
    # df_cell is a DataFrame containing cell-level averages and across-trial variability metrics
    
    agg_map = {
        "mean_firing_rate_movie": ("firing_rate_movie", "mean"),
        "firing_rate_variance_across_trials": ("firing_rate_movie", safe_var),
        "mean_ISI_movie": ("ISI_mean_movie", "mean"),
        "mean_ISI_var_movie": ("ISI_var_movie", "mean"),
        "mean_ISI_std_movie": ("ISI_std_movie", "mean"),
        "mean_CV_ISI_movie": ("CV_ISI_movie", "mean"),
        "mean_voltage_movie": ("Vm_mean_movie", "mean"),
        "voltage_variance_movie": ("Vm_var_movie", "mean"),
        "n_trials": ("trial_id", "count")}

    df_cell = df_trial.groupby(["cell_id", "group"], as_index=False).agg(**agg_map)

    return df_cell

def run_movie_current_clamp_analysis(filename, duration_s=42, prominence=42, distance_ms=10, pre_grey_s=(0, 4), movie_s=(4, 39), post_grey_s=(39, 42), condition_map=None, metrics_to_compare=None):
    # runs the full movie current-clamp analysis pipeline
    
    # input
    # filename is the path to the MATLAB .mat file
    # duration_s is the total recording duration in seconds
    # prominence is the minimum spike prominence used for spike detection
    # distance_ms is the minimum distance between detected spikes in ms
    # pre_grey_s is a tuple with the start and end time of the pre-movie grey screen in seconds
    # movie_s is a tuple with the start and end time of the movie stimulus in seconds
    # post_grey_s is a tuple with the start and end time of the post-movie grey screen in seconds
    # condition_map is a dictionary mapping group labels to condition names
    # metrics_to_compare is a list of cell-level metrics used for group statistics
    
    # output
    # results is a dictionary containing raw data, traces, metadata, time vectors, metrics, summaries, and statistics
    
    if condition_map is None:
        condition_map = {1: "CTR", 2: "FR"}

    if metrics_to_compare is None:
        metrics_to_compare = ["mean_firing_rate_movie", "firing_rate_variance_across_trials", "mean_ISI_movie", "mean_ISI_var_movie", "mean_ISI_std_movie", "mean_CV_ISI_movie", "mean_voltage_movie", "voltage_variance_movie"]

    traces_raw, groups_raw = load_movie_current_clamp_data(filename)
    traces, meta = unpack_movie_current_clamp(traces_raw, groups_raw)
    time, fs_eff, pre_grey_idx, movie_idx, post_grey_idx = make_time_and_epoch_indices(traces.shape[1], duration_s=duration_s, pre_grey_s=pre_grey_s, movie_s=movie_s, post_grey_s=post_grey_s)
    traces_by_group, meta_by_group = split_traces_by_group(traces, meta, group_values=tuple(condition_map.keys()))
    df_trial = compute_trial_metrics(traces, meta, time, movie_idx, prominence=prominence, distance_ms=distance_ms)
    df_cell = aggregate_cell_metrics(df_trial)
    
    return df_cell

def save_firing_statistics_to_pickle(df_cell, save_path="firing_statistics.pkl"):
    # extract firing statistics from df_cell and save them to one pickle file
    
    # input
    # df_cell is a pandas dataframe containing firing statistics for each cell
    # save_path is the path where the pickle file is saved
    
    # output
    # firing_statistics is a dictionary containing firing rate, ISI variance and CV_ISI values for CTR and FR
    
    firing_statistics = {
        "firing_rate_movie_CTR": df_cell.loc[df_cell["group"] == 1, "mean_firing_rate_movie"].dropna().to_numpy(),
        "firing_rate_movie_FR": df_cell.loc[df_cell["group"] == 2, "mean_firing_rate_movie"].dropna().to_numpy(),
        "ISI_var_movie_CTR": df_cell.loc[df_cell["group"] == 1, "mean_ISI_var_movie"].dropna().to_numpy(),
        "ISI_var_movie_FR": df_cell.loc[df_cell["group"] == 2, "mean_ISI_var_movie"].dropna().to_numpy(),
        "CV_ISI_movie_CTR": df_cell.loc[df_cell["group"] == 1, "mean_CV_ISI_movie"].dropna().to_numpy(),
        "CV_ISI_movie_FR": df_cell.loc[df_cell["group"] == 2, "mean_CV_ISI_movie"].dropna().to_numpy()}
    
    with open(save_path, "wb") as f:
        pickle.dump(firing_statistics, f)
    
    print(f"Saved firing statistics to: {save_path}")
    
    return firing_statistics





