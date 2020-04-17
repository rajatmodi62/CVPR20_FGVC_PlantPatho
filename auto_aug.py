from auto_aug_utils.utils import (sample_policy)
from auto_aug_utils.writer import (Writer)
from train import train
from utils.print_util import cprint

TOP_POLICY_CNT = 5
SUB_POLICY_CNT = 5
OPERATIONS_PER_SUB = 2
AUG_SEARCH_LOOP = 3


def search(config, device):
    writer = Writer()

    cprint("[ Getting baseline policy ]", type="info2")
    # get a baseline score with no augmentations
    best_loss, best_roc = train(config, device, policy=None)
    writer.write(-1, best_roc)
    cprint("[ Baseline score: " + str(best_roc) + " ]", type="success")

    policies = []

    # augmentation search loop
    for i in range(AUG_SEARCH_LOOP):
        cprint("[ Policy search Iteration : " + str(i + 1) +
               "/" + str(AUG_SEARCH_LOOP) + " ]", type="info2")

        # sample a policy from pool
        policy = sample_policy(SUB_POLICY_CNT, OPERATIONS_PER_SUB)

        # get best score using the policy
        best_loss, best_roc = train(config, device, policy=policy)
        writer.write(i, best_roc)

        # add policy to list
        policies.append((best_roc, policy))

        # sort based on score and pick best 5
        policies = list(
            sorted(policies, key=lambda v: v[0]))[-1 * TOP_POLICY_CNT:]

        # save policy list
        writer.freeze_policies(policies)

        cprint("[ Policy Score: " + str(best_roc) + " ]", type="success")

    pass
