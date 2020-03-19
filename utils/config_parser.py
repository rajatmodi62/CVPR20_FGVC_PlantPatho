import yaml

def get_config_data( yml_path ):
    stream = open( yml_path, 'r' )
    config = yaml.safe_load(stream)
    return config