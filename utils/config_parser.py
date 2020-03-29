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

        # dataset (train)
        if 'train_dataset' not in config.keys():
            print('[ Training dataset not mentioned ]')
            exit()
        else:
            if 'name' not in config['train_dataset'].keys():
                config['train_dataset']['name'] = None
            if 'fold' not in config['train_dataset'].keys():
                config['train_dataset']['fold'] = None

        # dataset (val)
        if 'val_dataset' not in config.keys():
            print('[ Validation dataset not mentioned ]')
            exit()
        else:
            if 'name' not in config['val_dataset'].keys():
                config['val_dataset']['name'] = None
            if 'fold' not in config['val_dataset'].keys():
                config['val_dataset']['fold'] = None

        # model
        if 'model' not in config.keys():
            print('[ Model not mentioned ]')
            exit()
        else:
            if 'name' not in config['model'].keys():
                print('[ Model name not mentioned ]')
                exit()
            if 'pred_type' not in config['model'].keys():
                print('[ Prediction type not mentioned ]')
                exit()
            if config['model']['pred_type'] not in ['classification', ' regression', 'mixed']:
                print('[ Prediction type must be either of  classification/regression/mixed ]')
                exit()
            if 'num_classes' not in config['model'].keys():
                print('[ Number of classes not mentioned ]')
                exit()
            if 'tuning_type' not in config['model'].keys():
                print('[ Number of classes not mentioned ]')
                config['model']['tuning_type'] = None
            if 'hyper_params' not in config['model'].keys():
                config['model']['hyper_params'] = None

        # optimiser
        if 'optimiser' not in config.keys():
            print('[ Optimiser not mentioned ]')
            exit()
        else:
            if 'name' not in config['optimiser'].keys():
                print('[ Optimiser name not mentioned ]')
                exit()
            if 'hyper_params' not in config['optimiser'].keys():
                config['optimiser']['hyper_params'] = None

        # scheduler
        if 'scheduler' not in config.keys():
            config['scheduler'] = {
                "name": None,
                "hyper_params": None
            }
        else:
            if 'name' not in config['scheduler'].keys():
                print('[ Scheduler name not mentioned ]')
                exit()
            if 'hyper_params' not in config['scheduler'].keys():
                config['scheduler']['hyper_params'] = None

        # loss function
        if 'loss_function' not in config.keys():
            print('[ Loss function not mentioned ]')
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

        # dataset (test)
        if 'test_dataset' not in config.keys():
            print('[ Validation dataset not mentioned ]')
            exit()
        else:
            if 'name' not in config['test_dataset'].keys():
                config['test_dataset']['name'] = None

        # model list
        if 'model_list' not in config.keys():
            print('[ Model List not mentioned ]')
            exit()
        else:
            if len(config['model_list']) > 0:
                for model in config['model_list']:
                    if 'name' not in model['model'].keys():
                        print('[ Model name not mentioned ]')
                        exit()
                    if 'num_classes' not in model['model'].keys():
                        print('[ Number of classes not mentioned ]')
                        exit()
                    if 'path' not in model['model'].keys():
                        print('[ Model weights paths not mentioned ]')
                        exit()
                    if 'hyper_params' not in model['model'].keys():
                        model['model']['hyper_params'] = None
            else:
                print('[ No models in Model List ]')
                exit()

    return config


def get_config_data(yml_file_name):
    name = yml_file_name.split('.')[0]

    yml_path = path.join('config', yml_file_name)

    stream = open(yml_path, 'r')
    config = yaml.safe_load(stream)

    config['experiment_name'] = name

    return hydrate_config(config)
