# Read Me
These are codes for the experiments run in our paper "Better Algorithms for Individually Fair k-Clustering" in https://arxiv.org/abs/2106.12150


To start, note that there is a config file /config/priority_config.ini that we refer to from now on for setting parameters.

Download the datasets referenced to in the paper and add them to /data/ folder. For diabetes specifically, run /util/clean_diabetes_script.py to fix the age column.

Currently, the code is structured so that the processed and subsampled datasets are written to and read from /subsamples.

To create fresh subsamples, run clean_subsample_script.py and before that, remember to change the random seed in config. The random seed is set to 0 by default which is what we used to generate our subsamples.

All the experiments are coded in main.py. To select the objective function, the dataset, the number of clusters, and other parameters, make changes to the config file before running main.py. Currently it is structured to log the output into output/kmeans/. Refer to the documentation in the code to see what parameters are logged. The same fixed random seed is used to produce the same output as what we have in the paper.

For the cost of fairness experiments, there is a separate script called cost_of_fairness.py that saves the output into a folder named cost_of_fairness inside of the specified output directory from the config file.

To plot all experiments, run make_plots_script.py. All the plots will be saved in the same directory as the experiment output directory specified in config. Currently, it only plots figures used in the paper but there are many other stats that we log and the reader might be interested to see. There are examples commented out in the plot script.

