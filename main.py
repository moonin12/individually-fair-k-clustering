import configparser
from util.configutil import read_list
from util.data_process_utils import get_client_neighborhood_graph, get_clustering_stats
from priority_lp_solver import solve_priority_p_k_clustering_lp
from clustering_algos import fair_round, sparsify_and_solve_lp, mahabadi_vakilian, jung_etal, arya_etal_driver, \
    kmeanspp_driver
import numpy as np
import random

from util.read_write_utils import read_subsampled_data, write_output
from util.data_process_utils import get_client_neighborhood

# Read config file
config_file = "config/priority_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)

output_dir = config["main"].get("output_dir")
input_dir = config["main"].get("input_dir")
lp_method = int(config["main"].get("lp_method"))
rand_seed = int(config["main"].get("rand_seed"))
power = int(config["main"].get("power"))

# Set random seeds for both random and numpy
random.seed(rand_seed)
np.random.seed(rand_seed)

# Uncomment if you'd like the terminal output to be saved. Remember to close terminal_out in the end
# terminal_out = open(input_dir + "logs.out", 'w') #REMEMBER TO CLOSE
# sys.stdout = terminal_out

num_clusters = list(map(int, config["main"].getlist("num_clusters")))
dataset_name = config["main"].get("dataset")
sub_sample_numbers = list(map(int, config["main"].getlist("sub_sample_numbers")))
delta = float(config["main"].get("delta"))

for number in sub_sample_numbers:

    all_pair_distances, df = read_subsampled_data(config["main"].get("max_points"), input_dir, dataset_name, number)

    nr_points = len(df.index)

    for n_clusters in num_clusters:
        output = {}
        output["dataset_name"] = dataset_name
        output["nr_points"] = nr_points
        output["power"] = power
        output["k"] = n_clusters
        output["n_over_k"] = nr_points // n_clusters
        neighborhood, radii = get_client_neighborhood(nr_points, all_pair_distances, nr_points // n_clusters)

        output["neighborhood_radii"] = radii
        # output["neighborhood_points"] = neighborhood

        # ------------ Solving the LP ----------------
        print("------- starting to solve LP")
        lp_res = solve_priority_p_k_clustering_lp([1] * nr_points, radii, neighborhood, all_pair_distances, n_clusters,
                                                  power, lp_method)
        output["lp_solver_res"] = lp_res
        output["phi"] = np.power(np.sum(np.power(radii, power)), 1 / power) / lp_res["cost"]

        if power == 2:
            print("------- starting kmeans++")
            # Run k-means++ algo without fairness constraints, starting with random centers
            kmeanspp_output = kmeanspp_driver(df, n_clusters, 0.02, 1000, radii, rand_seed)
            kmeanspp_output["ratio_to_lp"] = kmeanspp_output["cost"] / lp_res["cost"]
            output["kmeans++_output"] = kmeanspp_output

        print("------- starting Arya et. al.")
        # Run Arya et. al.'s algo without fairness constraints, starting with random centers
        vanilla_output = arya_etal_driver(nr_points, all_pair_distances, n_clusters, power, 0.02, 1000, radii)
        vanilla_output["ratio_to_lp"] = vanilla_output["cost"] / lp_res["cost"]
        output["van_output"] = vanilla_output

        print("------- starting Jung Kannan Lutz")
        jkl_output = jung_etal(nr_points, all_pair_distances, n_clusters, power, radii)
        jkl_output["ratio_to_lp"] = jkl_output["cost"] / lp_res["cost"]
        output["jkl_output"] = jkl_output

        print("------- starting Mahabadi Vakilian")
        mv_output = mahabadi_vakilian(nr_points, all_pair_distances, n_clusters, power, radii)
        mv_output["ratio_to_lp"] = mv_output["cost"] / lp_res["cost"]
        output["mv_output"] = mv_output

        print("------- starting Fair Round")
        fr_output = fair_round(nr_points, all_pair_distances, n_clusters, power, radii, lp_res["y"],
                               lp_res["cost_shares"], do_binsearch=True)
        fr_output["time"] = fr_output["time"] + lp_res["time"]
        fr_output["ratio_to_lp"] = fr_output["cost"] / lp_res["cost"]
        output["fr_output"] = fr_output

        # ------------ Solving the sparsified LP ----------------
        print("------- starting to solve sparsified LP")
        spa_lp_res, lp_y, lp_cost_shares = sparsify_and_solve_lp(all_pair_distances, radii, n_clusters, power, delta,
                                                                 lp_method=lp_method)
        spa_lp_res["ratio_to_lp"] = spa_lp_res["sparse_lp_solver_res"]["cost"] / lp_res["cost"]
        output["spa_lp_solver_res"] = spa_lp_res

        print("------- starting Fair Round on Sparsified LP")
        spa_fr_output = fair_round(nr_points, all_pair_distances, n_clusters, power, radii, lp_y, lp_cost_shares,
                                   do_binsearch=True)
        spa_fr_output["time"] = spa_fr_output["time"] + spa_lp_res["sparse_lp_solver_res"]["time"]
        spa_fr_output["ratio_to_lp"] = spa_fr_output["cost"] / spa_lp_res["sparse_lp_solver_res"]["cost"]
        output["spa_fr_output"] = spa_fr_output

        # # ------------- allowing the same violations as MV -----------#
        mv_relaxed_radii = np.maximum(mv_output["dist_vec"], radii).tolist()
        output["mv_relaxed_neighborhood_radii"] = mv_relaxed_radii
        mv_relaxed_neighborhood = get_client_neighborhood_graph(nr_points, all_pair_distances, mv_relaxed_radii)

        print("------- starting to solve mv-relaxed LP")
        mv_lp_res = solve_priority_p_k_clustering_lp([1] * nr_points, mv_relaxed_radii, mv_relaxed_neighborhood,
                                                     all_pair_distances, n_clusters,
                                                     power, lp_method)
        mv_lp_res["ratio_to_lp"] = mv_lp_res["cost"] / lp_res["cost"]
        output["mv_lp_solver_res"] = mv_lp_res

        print("------- starting Fair Round on mv_relaxed LP")
        mv_fr_output = fair_round(nr_points, all_pair_distances, n_clusters, power, mv_relaxed_radii, mv_lp_res["y"],
                                  mv_lp_res["cost_shares"], do_binsearch=True)
        mv_fr_output["time"] = mv_fr_output["time"] + mv_lp_res["time"]

        # WARNING: have to fix the reported output to reflect violations w.r.t. the original radii
        _, mv_rel_radii_violations, _, mv_rel_max_violation, mv_rel_nr_fair = get_clustering_stats(nr_points,
                                                                                                   all_pair_distances,
                                                                                                   radii,
                                                                                                   mv_fr_output[
                                                                                                       "centers"],
                                                                                                   power)
        mv_fr_output["radii_violations"] = mv_rel_radii_violations.tolist()
        mv_fr_output["max_violation"] = mv_rel_max_violation
        mv_fr_output["nr_fair"] = mv_rel_nr_fair
        mv_fr_output["ratio_to_lp"] = mv_fr_output["cost"] / lp_res["cost"]
        output["mv_fr_output"] = mv_fr_output

        write_output(output, output_dir, dataset_name, nr_points, number, n_clusters)

# terminal_out.close()
