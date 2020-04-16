from auto_aug_utils.utils import (sample_policy)
from auto_aug_utils.writer import (Writer)
from train import train

TOP_POLICY_CNT = 5
SUB_POLICY_CNT = 5
OPERATIONS_PER_SUB = 2


def search(config, device):
    writer = Writer()

    # get a baseline score with no augmentations
    score_1, score_2 = train(config, device, transform=None)
    writer.write(-1, score_2)

    print("[ Base Score: ", score_2, " ]")

    policies = []

    # augmentation search loop
    for i in range(50):
        # sample a policy from pool
        policy = sample_policy(SUB_POLICY_CNT, OPERATIONS_PER_SUB)

        # get best score using the policy
        score_1, score_2 = train(config, device, transform=policy)
        writer.write(i, score_2)

        # add policy to list
        policies.append((score_2, policy))

        # sort based on score and pick best 5
        policies = list(
            sorted(policies, key=lambda v: v[0]))[-1 * TOP_POLICY_CNT:]

        # save policy list
        writer.freeze_policies(policies)

    pass
