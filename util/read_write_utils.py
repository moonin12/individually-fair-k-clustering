import pandas as pd
import numpy as np
import json

from scipy.spatial.distance import pdist, squareform
from util.old_read_data_utils import subsample_data, read_data, clean_data


# Function that subsamples a given dataset and writes: a subsampled dataset, along with all pairs distances in the sample
# Input:
#   df: dataset (as a dataframe)
#   sample_size: number of points to sample from the dataset
#   data_dir: where to save the output
#   name: the dataset's name
#   count: total number of subsamples needed
# Output: For subsample i (from 0 to count-1) write to disk:
#   data_file: The subsample in <datadir>/<name>_<sample_size>_<i>.csv
#   distances_file: matrix of all pairs distances loaded from <datadir>/<name>_<sample_size>_<number>_distances.csv
def write_subsampled_data(df, sample_size, data_dir, name, count):
    for i in range(count):
        subsample = subsample_data(df, sample_size)
        all_pair_distances = pd.DataFrame(squareform(pdist(subsample.values, 'euclidean')))

        address_string = data_dir + name + "_" + str(sample_size) + "_" + str(i)
        data_file = address_string + ".csv"
        distances_file = address_string + "_distances.csv"

        subsample.to_csv(data_file, index=False)
        all_pair_distances.to_csv(distances_file, index=False)


# Function that reads previously written subsamples
# Input:
#   sample_size: size of the subsample
#   data_dir: where to save the output
#   name: the dataset's name
#   number: index of the subsample needed
# Output:
#   all_pairs_distances: matrix of all pairs distances loaded from <datadir>/<name>_<sample_size>_<number>_distances.csv
#   df: dataset (as a dataframe) loaded from <datadir>/<name>_<sample_size>_<number>.csv
def read_subsampled_data(sample_size, data_dir, name, number):
    address_string = data_dir + name + "_" + str(sample_size) + "_" + str(number)
    data_file = address_string + ".csv"
    distances_file = address_string + "_distances.csv"
    df = pd.read_csv(data_file)
    all_pairs_distances = pd.read_csv(distances_file)
    return all_pairs_distances.values, df


# Function that reads data, cleans data, generates subsamples and writes them to disk according to config
def create_subsampled_data(config):
    dataset_name = config["main"].get("dataset")
    input_dir = config["main"].get("input_dir")
    rand_seed = int(config["main"].get("rand_seed"))

    # Set random seed for numpy
    np.random.seed(rand_seed)

    # Read data in from a given csv_file found in config
    df = read_data(config, dataset_name)
    # Clean the data (bucketize text data)
    df = clean_data(df, config, dataset_name)

    max_points = config["main"].getint("max_points")
    nr_sub_samples = config["main"].getint("nr_sub_samples")

    # Subsample data if needed
    if max_points and len(df) > max_points:
        write_subsampled_data(df, max_points, input_dir, dataset_name, nr_sub_samples)


# Function that writes a dictionary (outpout of an experiment) as a json file
# Input:
#   output: output dictionary to be written
#   data_dir: where to save the output
#   name: the dataset's name
#   count: total number of points in the dataset
#   number: the experiment number, assumed to be the same as the subsample number used
#   n_clusters: number of clusters
#   suffix: allowing arbitrary suffix for the output
# Output:
#   writes output to <datadir>/<name>_<count>_<number>_k_<n_clusters>.json
def write_output(output, data_dir, name, count, number, n_clusters, suffix=""):
    address_string = data_dir + name + "_" + str(count) + "_" + str(number)
    data_file = address_string + "_k_" + str(n_clusters) + suffix + ".json"
    with open(data_file, "w") as dataf:
        json.dump(output, dataf)


# Function that reads a json file corresponding to a previous experiment output
# Input:
#   data_dir: where to save the output
#   name: the dataset's name
#   count: total number of points in the dataset
#   number: the experiment number, assumed to be the same as the subsample number used
#   n_clusters: number of clusters
# Output:
#   reads file from <datadir>/<name>_<count>_<number>_k_<n_clusters>.json
def read_output(data_dir, name, count, number, n_clusters):
    address_string = data_dir + name + "_" + str(count) + "_" + str(number)
    data_file = address_string + "_k_" + str(n_clusters) + ".json"
    return json.load(data_file)
