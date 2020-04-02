from os import path
import yaml


def hydrate_config(config):
    if 'mode' not in config.keys():
        config['mode'] = None

    if config['mode'] == 'train':
        # basics
        if 'validation_frequency' not in config.keys():
            config['validation_frequency'] = None
        if 'epochs' not in config.keys():
            config['epochs'] = 10
        if 'batch_size' not in config.keys():
            config['batch_size'] = 1
        if 'num_classes' not in config.keys():
            print('[ Number of classes not mentioned ]')
            exit()

        # dataset (train)
        if 'train_dataset' not in config.keys():
            print('[ Training dataset key missing ]')
            exit()
        else:
            if 'name' not in config['train_dataset'].keys():
                print('[ Training dataset name not mentioned ]')
                exit()
            if 'fold' not in config['train_dataset'].keys():
                config['train_dataset']['fold'] = None

        # dataset (val)
        if 'val_dataset' not in config.keys():
            print('[ Validation dataset key missing ]')
            exit()
        else:
            if 'name' not in config['val_dataset'].keys():
                print('[ Validation dataset name not mentioned ]')
                exit()
            if 'fold' not in config['val_dataset'].keys():
                config['val_dataset']['fold'] = None

        # model
        if 'model' not in config.keys():
            print('[ Model key missing ]')
            exit()
        else:
            if 'name' not in config['model'].keys():
                print('[ Model name not mentioned ]')
                exit()
            if 'pred_type' not in config['model'].keys():
                print('[ Prediction type not mentioned ]')
                exit()
            if config['model']['pred_type'] not in ['classification', 'regression', 'mixed']:
                print(
                    '[ Prediction type must be either of classification/regression/mixed ]')
                exit()
            if 'tuning_type' not in config['model'].keys():
                config['model']['tuning_type'] = None
            else:
                print( config["model"]["tuning_type"] )
                if config["model"]["tuning_type"] not in ["fine-tuning", "feature-extraction"]:
                    print("[ Tuning type can be one of fine-tuning/feature-extraction ]")
                    exit()
            if 'hyper_params' not in config['model'].keys():
                config['model']['hyper_params'] = None
            if 'pre_trained_path' not in config['model'].keys():
                config['model']['pre_trained_path'] = None

        # optimiser
        if 'optimiser' not in config.keys():
            print('[ Optimiser key missing ]')
            exit()
        else:
            if 'name' not in config['optimiser'].keys():
                print('[ Optimiser name not mentioned ]')
                exit()
            if 'hyper_params' not in config['optimiser'].keys():
                config['optimiser']['hyper_params'] = None

        # scheduler
        if 'scheduler' not in config.keys():
            config['scheduler'] = None
        else:
            if 'name' not in config['scheduler'].keys():
                print('[ Scheduler name not mentioned ]')
                exit()
            if 'hyper_params' not in config['scheduler'].keys():
                config['scheduler']['hyper_params'] = None

        # loss function
        if 'loss_function' not in config.keys():
            print('[ Loss function key missing ]')
            exit()
        else:
            if 'name' not in config['loss_function'].keys():
                print('[ Loss function name not mentioned ]')
                exit()
            if 'hyper_params' not in config['loss_function'].keys():
                config['loss_function']['hyper_params'] = None

    elif config['mode'] == 'test':
        # basic
        if 'ensemble' not in config.keys():
            config['ensemble'] = False
        if 'num_classes' not in config.keys():
            print('[ Number of classes not mentioned ]')
            exit()

        # dataset (test)
        if 'test_dataset' not in config.keys():
            print('[ Test dataset key missing ]')
            exit()
        else:
            if 'name' not in config['test_dataset'].keys():
                print('[ Test dataset name not mentioned ]')
                exit()

        # model list
        if 'experiment_list' not in config.keys():
            print('[ Experiment List key missing ]')
            exit()
        else:
            if len(config['experiment_list']) > 0:
                for experiment in config['experiment_list']:
                    if 'path' not in experiment['experiment'].keys():
                        print('[ Experiment output path not mentioned ]')
                        exit()
                    hydrate_secondary_config(
                        experiment['experiment']['path'],
                        experiment['experiment']
                    )
            else:
                print('[ No experiment in Experiment List ]')
                exit()

    return config


def hydrate_secondary_config(yml_file_name, config):
    secondary_yml_path = None
    if path.exists(path.join('config', yml_file_name + '.yml')):
        secondary_yml_path = path.join('config', yml_file_name + '.yml')
    else:
        print("[ Experiment file missing from config ]")
        exit()

    stream = open(secondary_yml_path, 'r')
    secondary_config = yaml.safe_load(stream)
    secondary_config = hydrate_config(secondary_config)

    config['name'] = secondary_config['model']['name']
    config['pred_type'] = secondary_config['model']['pred_type']
    config['hyper_params'] = secondary_config['model']['hyper_params']


def get_config_data(yml_file_name):
    name = yml_file_name.split('.')[0]

    yml_path = path.join('config', yml_file_name)

    stream = open(yml_path, 'r')
    config = yaml.safe_load(stream)

    config['experiment_name'] = name

    return hydrate_config(config)
