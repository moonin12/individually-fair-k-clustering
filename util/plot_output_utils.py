import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import json


def histogram_experiment_series(files, benchmark_styles, benchmarks, value_name, group_by_name, plot_y_label,
                                plot_x_label, plot_title,
                                data_dir, plot_name):
    # print("Experiments related to {} are {}".format(dataset_name, files))

    values = dict()
    for benchmark in benchmarks:
        values[benchmark] = dict()
    keys = set()
    sub_sample_nums = set()

    for sub_sample_num, file_list in files.items():
        # print("subsample_num = {}".format(sub_sample_num))
        # print("file_list = {}".format(file_list))
        for file in file_list:
            # print("reading from {}".format(data_dir + file))
            opened_file = open(data_dir + file, )
            output = json.loads(opened_file.read())
            opened_file.close()

            key = output[group_by_name]
            keys.add(key)
            sub_sample_nums.add(sub_sample_num)

            for benchmark in benchmarks:
                temp_dict = output[benchmark_styles[benchmark][2]][value_name]

                # print("key {}, sub_num {}, val_name {}".format(key, sub_sample_num, benchmark))
                # print("temp dict {}".format(temp_dict))
                try:
                    values[benchmark][sub_sample_num][key] = list(temp_dict)
                except:
                    values[benchmark][sub_sample_num] = dict()
                    values[benchmark][sub_sample_num][key] = list(temp_dict)

    # Uncomment if you want plots for each value of k separately
    # for sub_sample_num in sub_sample_nums:
    #     fig, ax = plt.subplots()
    #     bins = [0.1 * i for i in range(1, 31)]
    #     for key in keys:
    #         for benchmark in benchmarks:
    #             value_array = values[benchmark][sub_sample_num][key]
    #             plt.hist(value_array,
    #                      bins=bins,
    #                      density=False,
    #                      alpha=0.5,
    #                      label=benchmark,
    #                      color=benchmark_styles[benchmark][1])
    #
    #         # plt.legend(loc='upper right')
    #         # lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #         lgd = plt.legend(loc='upper right', fontsize=18)
    #         plt.ylabel(plot_y_label, fontsize=18)
    #         plt.xlabel(plot_x_label, fontsize=18)
    #         plt.title(plot_title, fontsize=18)
    #
    #         plt.savefig(data_dir + plot_name + "_" + str(sub_sample_num) + "_" + str(key) + ".png",
    #                     bbox_extra_artists=(lgd,), bbox_inches='tight')
    #         plt.clf()

    fig, ax = plt.subplots()
    bins = [0.1 * i for i in range(1, 21)]
    # min_fairness = 1000
    for key in keys:
        for benchmark in benchmarks:
            # print("SUBSAMP NUM {}".format(sub_sample_nums))
            value_array = [values[benchmark][sub_sample_num][key] for sub_sample_num in sub_sample_nums]
            value_avg = np.average(value_array, axis=0)

            y, bin_edges, _ = plt.hist(value_avg,
                                       bins=bins,
                                       density=False,
                                       alpha=0.5,
                                       label=benchmark,
                                       color=benchmark_styles[benchmark][1])

        plt.legend(loc='upper right', fontsize=18)
        plt.ylabel(plot_y_label, fontsize=18)
        plt.xlabel(plot_x_label, fontsize=18)
        plt.title(plot_title + " (k = " + str(key) + ")", fontsize=18)

        plt.savefig(data_dir + plot_name + "_aggr_" + str(key) + ".png")
        plt.clf()

    # print("completely fair on this percent: {}".format(min_fairness / 1000))


def line_plot_experiment_series(files, benchmark_styles, benchmarks, value_name, key_name, plot_y_label, plot_x_label,
                                plot_title, data_dir,
                                plot_name):
    # print("Experiments related to {} are {}".format(dataset_name, files))

    keys = set()
    sub_sample_nums = set()

    values = dict()
    for benchmark in benchmarks:
        values[benchmark] = dict()

    for sub_sample_num, file_list in files.items():
        # print("subsample_num = {}".format(sub_sample_num))
        # print("file_list = {}".format(file_list))
        for file in file_list:
            # print("reading from {}".format(data_dir + file))
            opened_file = open(data_dir + file, )
            output = json.loads(opened_file.read())
            opened_file.close()

            key = output[key_name]
            keys.add(key)
            sub_sample_nums.add(sub_sample_num)

            for benchmark in benchmarks:

                temp_dict = output[benchmark_styles[benchmark][2]][value_name]

                v = float(temp_dict)

                try:
                    values[benchmark][sub_sample_num][key] = v
                except:
                    values[benchmark][sub_sample_num] = dict()
                    values[benchmark][sub_sample_num][key] = v

    keys = sorted(keys)

    for sub_sample_num in sub_sample_nums:
        fig, ax = plt.subplots()
        for benchmark in benchmarks:
            value_array = [values[benchmark][sub_sample_num][key] for key in keys]
            ax.plot(keys, value_array, benchmark_styles[benchmark][0], label=benchmark, markersize=12)

        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
        plt.xticks(keys, keys)

        plt.ylabel(plot_y_label, fontsize=18)
        plt.xlabel(plot_x_label, fontsize=18)
        plt.title(plot_title, fontsize=18)

        plt.savefig(data_dir + plot_name + "_" + str(sub_sample_num) + ".png", bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.clf()

    fig, ax = plt.subplots()
    for benchmark in benchmarks:
        # print("SUBSAMP NUM {}".format(sub_sample_nums))
        value_array = [[values[benchmark][sub_sample_num][key] for key in keys] for sub_sample_num in sub_sample_nums]
        value_avg = np.average(value_array, axis=0)
        # print("value array {}".format(value_array))
        value_std = np.std(value_array, axis=0)
        # print("value avg {}".format(value_avg))
        # print("value std {}".format(value_std))
        # print("keys {}".format(keys))
        ax.errorbar(keys, value_avg, fmt=benchmark_styles[benchmark][0], yerr=value_std, label=benchmark, capsize=6,
                    markersize=12)

    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
    plt.xticks(keys, keys)

    plt.ylabel(plot_y_label, fontsize=18)
    plt.xlabel(plot_x_label, fontsize=18)
    plt.title(plot_title, fontsize=18)

    plt.savefig(data_dir + plot_name + "_aggr" + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()


# Function that reads the files in the data_dir and groups them based on the dataset name
# Input:
#   data_dir: where to read output files from
def read_all_output(data_dir):
    files = listdir(data_dir)
    # print("The first {} and the last {}".format(files[0], files[-1]))
    # print("files are {}".format(files))
    file_by_dataset = {}
    for file in files:
        split_type = file.split('.')
        if len(split_type) < 2 or split_type[1] != "json": continue

        split_name = file.split('_')
        dataset_name = split_name[0] + "_" + split_name[1]
        dataset_subsamp_num = int(split_name[2])

        if dataset_name in file_by_dataset.keys():
            if dataset_subsamp_num in file_by_dataset[dataset_name].keys():
                file_by_dataset[dataset_name][dataset_subsamp_num].append(file)
            else:
                file_by_dataset[dataset_name][dataset_subsamp_num] = [file]
        else:
            file_by_dataset[dataset_name] = {}
            file_by_dataset[dataset_name][dataset_subsamp_num] = [file]

    return file_by_dataset
