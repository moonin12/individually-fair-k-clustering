from util.configutil import read_list
import configparser
from util.plot_output_utils import read_all_output, line_plot_experiment_series, histogram_experiment_series

config_file = "config/priority_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)
data_dir = config["main"].get("output_dir")
power = int(config["main"].get("power"))


def benchmark_comparision_plots():
    file_by_dataset = read_all_output(data_dir)

    key_name = "k"
    styles = {"Fair-Round": ["-bo", "blue", "fr_output"],
              "Sparse Fair-Round": ["-gv", "green", "spa_fr_output"],
              "Fair LP": ["-k*", "black", "lp_solver_res"],

              "MV": ["-r^", "red", "mv_output"],
              "MV-relaxed LP": ["-kp", "black", "mv_lp_solver_res"],
              "MV-Relaxed Fair-Round": ["-gv", "green", "mv_fr_output"],

              "JKL": ["-yH", "yellow", "jkl_output"],

              "Arya et. al.": ["-cD", "cyan", "van_output"],
              "k-means++": ["-md", "magenta", "kmeans++_output"]
              }

    # List of all available benchmarks: ["Fair-Round", "Sparse Fair-Round", "Fair LP", "MV", "MV-relaxed LP", "MV-Relaxed Fair-Round", "JKL", "Arya et. al.", "k-means++"]
    for dataset_name in file_by_dataset.keys():
        split_name = dataset_name.split('_')
        title = split_name[0]

        fairness_benchmarks = ["Fair-Round", "MV", "JKL"]
        line_plot_experiment_series(file_by_dataset[dataset_name], styles, fairness_benchmarks, "max_violation",
                                    key_name, "Maximum fairness violation",
                                    "Number of clusters", title, data_dir, dataset_name + "_maxviolations")

        # nr_fair_benchmarks = ["Fair-Round", "MV", "JKL"]
        # line_plot_experiment_series(file_by_dataset[dataset_name], styles, nr_fair_benchmarks, "nr_fair", key_name,
        #                             "Number of satisfied points",
        #                             "Number of clusters", title, data_dir, dataset_name + "_nr_fair")

        cost_benchmarks = ["Fair-Round", "Fair LP", "MV", "JKL"]
        line_plot_experiment_series(file_by_dataset[dataset_name], styles, cost_benchmarks, "cost", key_name, "Costs",
                                    "Number of clusters", title, data_dir, dataset_name + "_costs")

        # cost_ratio_to_lp_benchmarks = ["Fair-Round", "MV", "JKL", "MV-Relaxed Fair-Round"]
        # line_plot_experiment_series(file_by_dataset[dataset_name], styles, cost_ratio_to_lp_benchmarks, "cost", key_name, "Costs divided by LP cost",
        #                             "Number of clusters", title, data_dir, dataset_name + "_ratio_to_lp_costs")

        runtime_benchmarks = ["Fair-Round", "Sparse Fair-Round", "MV", "JKL"]
        if power == 2: runtime_benchmarks.append("k-means++")
        line_plot_experiment_series(file_by_dataset[dataset_name], styles, runtime_benchmarks, "time", key_name,
                                    "Runtime in seconds",
                                    "Number of clusters", title, data_dir, dataset_name + "_times")

        mv_fairness_hist_benchmarks = ["Fair-Round", "MV"]
        histogram_experiment_series(file_by_dataset[dataset_name], styles, mv_fairness_hist_benchmarks, "radii_violations",
                                    key_name, "Number of points in bins",
                                    "Fairness violation", "Fair-Round vs MV",
                                    data_dir, dataset_name + "_mv_radii_violations")

        jkl_fairness_hist_benchmarks = ["Fair-Round", "JKL"]
        histogram_experiment_series(file_by_dataset[dataset_name], styles, jkl_fairness_hist_benchmarks,
                                    "radii_violations",
                                    key_name, "Number of points in bins",
                                    "Fairness violation", "Fair-Round vs JKL",
                                    data_dir, dataset_name + "_jkl_radii_violations")

        rel_fairness_benchmarks = ["MV", "MV-Relaxed Fair-Round"]
        line_plot_experiment_series(file_by_dataset[dataset_name], styles, rel_fairness_benchmarks, "max_violation",
                                    key_name, "Maximum fairness violation",
                                    "Number of clusters", title, data_dir, dataset_name + "_rel_maxviolations")

        rel_cost_benchmarks = ["Fair LP", "MV", "MV-relaxed LP", "MV-Relaxed Fair-Round"]
        line_plot_experiment_series(file_by_dataset[dataset_name], styles, rel_cost_benchmarks, "cost", key_name, "Costs",
                                    "Number of clusters", title, data_dir, dataset_name + "_rel_costs")

        spa_fairness_benchmarks = ["Fair-Round", "Sparse Fair-Round", "MV", "JKL"]
        line_plot_experiment_series(file_by_dataset[dataset_name], styles, spa_fairness_benchmarks, "max_violation",
                                    key_name, "Maximum fairness violation",
                                    "Number of clusters", title, data_dir, dataset_name + "_spa_maxviolations")

        spa_cost_benchmarks = ["Fair-Round", "Sparse Fair-Round", "MV", "JKL"]
        line_plot_experiment_series(file_by_dataset[dataset_name], styles, spa_cost_benchmarks, "cost", key_name, "Costs",
                                    "Number of clusters", title, data_dir, dataset_name + "_spa_costs")


def cost_of_fairness_plots():
    dir = data_dir + "cost_of_fairness/"
    file_by_dataset = read_all_output(dir)

    key_name = "radii_dilation"
    style = {"Fair LP": ["k", "black", "lp_solver_res"]}

    # print("---- files by dataset {}".format(file_by_dataset))

    # List of all available benchmarks: ["Fair-Round", "Sparse Fair-Round", "Fair LP", "MV", "MV-relaxed LP", "MV-Relaxed Fair-Round", "JKL", "Arya et. al.", "k-means++"]
    for dataset_name in file_by_dataset.keys():

        files_by_curr_name = file_by_dataset[dataset_name]
        files_by_k_subsamp = {}

        # print("files by curr name {}".format(files_by_curr_name))
        for subsample in files_by_curr_name.keys():
            # files_by_k = {}
            for file in files_by_curr_name[subsample]:

                split_name = file.split('_')
                # print("split name {}".format(split_name))
                k = int(split_name[4].split('.')[0])

                # if k in files_by_k.keys():
                #     files_by_k[k].append(file)
                # else:
                #     files_by_k[k] = [file]

                if k not in files_by_k_subsamp.keys():
                    files_by_k_subsamp[k] = {}
                if subsample in files_by_k_subsamp[k].keys():
                    files_by_k_subsamp[k][subsample].append(file)
                else:
                    files_by_k_subsamp[k][subsample] = [file]

        # print(files_by_k_subsamp)
        for k in files_by_k_subsamp.keys():
            line_plot_experiment_series(files_by_k_subsamp[k], style, ["Fair LP"], "cost",
                                        key_name, "Fair LP cost",
                                        "Radii dilation", " Cost of fairness" + "(k = " + str(k)+")", dir,
                                        dataset_name + "_" + str(k) + "_costoffairness")


benchmark_comparision_plots()
cost_of_fairness_plots()
