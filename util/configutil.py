'''
From https://github.com/nicolasjulioflores/fair_algorithms_for_clustering
Functions that help with reading from a config file.
'''


# Reads the given config string in as a list
#   Allows for multi-line lists.
def read_list(config_string, delimiter=','):
    config_list = config_string.replace("\n", "").split(delimiter)
    return [s.strip() for s in config_list]


# Read a given range from config string in as a list
def read_range(config_string, delimiter=','):
    start, end, step = tuple(map(int, config_string.split(delimiter)))
    return list(range(start, end, step))
