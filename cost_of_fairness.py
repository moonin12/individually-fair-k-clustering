import configparser
from util.configutil import read_list
from priority_lp_solver import solve_priority_p_k_clustering_lp
from util.read_write_utils import read_subsampled_data, write_output
from util.data_process_utils import get_client_neighborhood, get_client_neighborhood_graph
import numpy as np

# Read config file
config_file = "config/priority_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)

output_dir = output_dir = config["main"].get("output_dir") + "cost_of_fairness/"
input_dir = config["main"].get("input_dir")
lp_method = int(config["main"].get("lp_method"))
power = int(config["main"].get("power"))

dataset_name = config["main"].get("dataset")
sub_sample_numbers = list(map(int, config["main"].getlist("sub_sample_numbers")))

for number in sub_sample_numbers:

    all_pair_distances, df = read_subsampled_data(config["main"].get("max_points"), input_dir, dataset_name, number)

    nr_points = len(df.index)

    radii_dilation = 0.5
    for n_clusters in [20]:
        while radii_dilation < 3:
            radii_dilation = radii_dilation + 0.5

            output = {}
            output["dataset_name"] = dataset_name
            output["nr_points"] = nr_points
            output["power"] = power
            output["k"] = n_clusters
            output["n_over_k"] = nr_points // n_clusters
            output["radii_dilation"] = radii_dilation

            _, radii = get_client_neighborhood(nr_points, all_pair_distances, nr_points // n_clusters)
            radii = np.multiply(radii, radii_dilation).tolist()
            neighborhood = get_client_neighborhood_graph(nr_points, all_pair_distances, radii)

            output["neighborhood_radii"] = radii
            # output["neighborhood_points"] = neighborhood

            # ------------ Solving the LP ----------------
            lp_res = solve_priority_p_k_clustering_lp([1] * nr_points, radii, neighborhood, all_pair_distances,
                                                      n_clusters,
                                                      power, lp_method)
            output["lp_solver_res"] = lp_res

            write_output(output, output_dir, dataset_name, nr_points, number, n_clusters, "_" + str(int(radii_dilation*10)))
