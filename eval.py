import torch
from torch.utils.data import DataLoader
from os import (path, environ)
from tqdm import (trange, tqdm)
from dataset.dataset_factory import DatasetFactory
from transformer.transformer_factory import TransformerFactory
from model.model_factory import ModelFactory
from utils.evaluation_utils import EvaluationHelper
from utils.print_util import cprint


def eval(config, device):
    # Create pipeline objects
    dataset_factory = DatasetFactory(org_data_dir='./data')

    transformer_factory = TransformerFactory()

    model_factory = ModelFactory()

    evaluation_helper = EvaluationHelper(
        config['experiment_name'],
        overwrite=True,
        ensemble=config['ensemble']
    )

    # =============================== Experiment  loop =================================

    for experiment_item in config['experiment_list']:
        cprint("[ Experiment : ", experiment_item['experiment']
               ['path'], " ]", type="info2")

        # ==================== Model testing / evaluation setup ========================

        test_dataset = dataset_factory.get_dataset(
            'test',
            config['test_dataset']['name'],
            transformer_factory.get_transformer(
                height=experiment_item['experiment']['resize_dims'],
                width=experiment_item['experiment']['resize_dims']
            )
        )

        model = model_factory.get_model(
            experiment_item['experiment']['name'],
            config['num_classes'],
            experiment_item['experiment']['pred_type'],
            experiment_item['experiment']['hyper_params'],
            None,
            experiment_item['experiment']['path'],
            experiment_item['experiment']['weight_type']
        ).to(device)

        # ===============================================================================

        # ===================== Model testing / evaluation  loop ========================

        model.eval()

        test_output_list = []
        for batch_ndx, sample in enumerate(tqdm(DataLoader(test_dataset, batch_size=1), desc="Samples : ")):
            with torch.no_grad():
                input = sample
                input = input.to(device)

                output = model.forward(input)

                test_output_list.append(output)

        test_output_list = torch.cat(test_output_list, dim=0)

        # use this list to write using a helper
        evaluation_helper.evaluate(
            experiment_item['experiment']['pred_type'],
            config['num_classes'],
            experiment_item['experiment']['path'],
            test_dataset.get_csv_path(),
            test_output_list
        )

        # ===============================================================================

    if config['ensemble']:
        evaluation_helper.ensemble(
            test_dataset.get_csv_path(),
            type="softmax"
        )

    # ===================================================================================
