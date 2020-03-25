from os import path
import yaml

def get_config_data( yml_file_name ):
    name = yml_file_name.split('.')[0]

    yml_path = path.join('config', yml_file_name)

    stream = open( yml_path, 'r' )
    config = yaml.safe_load(stream)

    config['experiment_name'] = name
    return config