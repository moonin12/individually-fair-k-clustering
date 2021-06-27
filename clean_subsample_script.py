import configparser
from util.configutil import read_list
from util.read_write_utils import create_subsampled_data

# Read config file
config_file = "config/priority_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)

create_subsampled_data(config)
