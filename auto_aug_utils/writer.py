from os import (makedirs, path)
from shutil import rmtree
from utils.print_util import cprint
import pandas as pd
import json


class Writer:
    def __init__(self, overwrite=True):
        if path.exists(path.join('auto_aug_utils/data')) == False:
            makedirs('auto_aug_utils/data')
        else:
            if overwrite:
                cprint("[ Auto Augment output exists - Overwriting! ]", type="warn")
                rmtree('auto_aug_utils/data')
                makedirs('auto_aug_utils/data')
            else:
                cprint(
                    "[ Auto Augment output exists - Manual deletion needed ]", type="warn")
                exit()

    def write(self, iter, score):
        df = pd.DataFrame(
            [[iter, score]])
        result_path = path.join(
            'auto_aug_utils/data', 'result.csv')

        if not path.isfile(result_path):
            df.to_csv(
                result_path,
                header=[
                    "Iteration", "Score"
                ],
                index=False
            )
        else:
            df.to_csv(result_path, mode='a', header=False, index=False)

    def freeze_policies(self, policies):
        filtered = [policy[1] for policy in policies]

        # save it to json
        with open('transformer/best_policy.json', 'w+') as fid:
            json_str = json.dumps(filtered, indent=2)
            fid.write(json_str)
