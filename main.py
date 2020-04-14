import argparse
from train import train
from eval import eval
from utils.config_parser import get_config_data
from utils.check_gpu import get_training_device

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("experiment_file",
                    help="The name of the experiment config file")
parser.add_argument('-p', '--publish', action='store_true', 
    help="publishes results to telegram")
args = parser.parse_args()

# Get experiment config values
if args.experiment_file is None:
    exit()
config = get_config_data(args.experiment_file, args.publish)

# Get GPU / CPU device instance
device = get_training_device()

if config['mode'] == 'test':
    eval( config, device )

elif config['mode'] == 'train':
    train( config, device )
else:
    print("[ Experiment Mode should either be train/test ]")
    exit()
