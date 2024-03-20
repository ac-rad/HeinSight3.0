import csv
from os import path

import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv

"""
This script is for finding the separation time after stirring is stopped
"""


def find_sep_time_from_volume(volume_path, vial_number, start_time, plot=False):
    df = read_csv(volume_path)
    time_series = df["Time"]

    # finding the index of the last 5 minutes
    max_time = time_series.iloc[-1]
    time_index = np.where(time_series > max_time - 5)[0]

    # find the vial data
    vial_volumes = f'vial {vial_number}'
    matching_columns = [col for col in df.columns if vial_volumes.lower() in col.lower()]

    # a collection of separation time of all segments
    sep_times = []
    for matching_column in matching_columns:
        volume = df[matching_column]
        # if not matching_column.endswith('1'):
        last_5_min = volume[time_index]
        plateau_min, plateau_max = min(last_5_min), max(last_5_min)
        noise = plateau_max - plateau_min
        plateau = (np.where(volume < plateau_min - noise) or np.where(volume > plateau_max + noise))[0]
        sep_time = time_series[plateau[-1] + 1] if len(plateau) > 0 else 0
        sep_times.append(sep_time)
        if plot:
            plt.plot(time_series, volume, label=matching_column)
    plateau_time = max(sep_times)
    if plot:
        plt.vlines(plateau_time, ymin=0, ymax=1, label="plateau", color='grey', linewidth=2, linestyle='dashed',
                   alpha=0.5)
        plt.legend(loc="upper right")
        plt.xlabel('Time (min)')
        plt.ylabel('Volume (mL)')
        plt.show()
    return plateau_time - start_time


def find_separation_time(npy_filepath, start_time, plot=False):
    turb = np.load(npy_filepath)
    time_series = [i / 1800 for i in range(0, turb.shape[0])]
    diffs_no_average = [0] + [sum((turb[i] - turb[i - 1]) ** 2) / turb.shape[1] for i in range(1, turb.shape[0])]

    # finding the index of the last 5 minutes
    last_5_index = turb.shape[0] - 5 * 1800
    last_5_min = diffs_no_average[last_5_index:]
    threshold = min(last_5_min) + max(last_5_min) * 2

    # find first above threshold value
    diffs_no_average = np.array(diffs_no_average)
    plateau = np.where(diffs_no_average > threshold)[0]
    plateau_time = (plateau[-1] + 1) / 1800

    if plot:
        plt.plot(time_series, diffs_no_average, label="Turbidity difference")
        plt.vlines(plateau_time, ymin=0, ymax=20, label="plateau", color='grey', linewidth=2, linestyle='dashed',
                   alpha=0.5)
        # plt.ylim(0, 1)
        # plt.xlim(10000,12000)
        plt.legend(loc="upper right")
        plt.xlabel('Time (min)')
        plt.ylabel('Turbidity difference')
        plt.show()
    return plateau_time - start_time


if __name__ == "__main__":
    trial_number = [1, 2, 3]
    vial_number = [1, 2, 3, 4, 5, 6]
    start_time = 4.54  # min
    output = []
    for trial in trial_number:
        folder_path = rf".\output\raw_data\trial_{trial}"
        volume_path = path.join(folder_path, "volumes.csv")
        separation_times = {}

        for vial in vial_number:
            sep_time_from_volume = find_sep_time_from_volume(volume_path, vial, start_time, trial == 1)
            turb_path = path.join(folder_path, f'turb_vial_{vial}.npy')
            separation_time = find_separation_time(turb_path, start_time, plot=trial == 1)
            separation_times[f"{vial}_turb"] = separation_time
            separation_times[f"{vial}_volume"] = sep_time_from_volume
        output.append(separation_times)

    with open("triplicate.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(separation_times.keys()))
        writer.writeheader()
        writer.writerows(output)
