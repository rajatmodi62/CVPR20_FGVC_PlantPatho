import random

op_options = [
    ('RandomContrast',    {'limit': (0.1, 0.3), 'p': [0.0, 0.25, 0.5, 0.75]}),
    ('RandomBrightness',  {'limit': (0.1, 0.3), 'p': [0.0, 0.25, 0.5, 0.75]}),
    ('RandomGamma',       {'gamma_limit': (
        (70, 90), (110, 130)), 'p': [0.0, 0.25, 0.5, 0.75]}),
    ('Blur',              {'blur_limit': (3, 5), 'p': [0.0, 0.25, 0.5, 0.75]}),
    ('MotionBlur',        {'blur_limit': (3, 5), 'p': [0.0, 0.25, 0.5, 0.75]}),
    ('InvertImg',         {'p': [0.0, 0.25, 0.5, 0.75]}),
    ('Rotate',            {'limit': (5, 45), 'p': [0.0, 0.25, 0.5, 0.75]}),
    ('ShiftScaleRotate',  {'shift_limit': (
        0.03, 0.12), 'scale_limit': 0.0, 'rotate_limit': 0, 'p': [0.0, 0.25, 0.5, 0.75]}),
    ('RandomScale',       {'scale_limit': (
        0.05, 0.20), 'p': [0.0, 0.25, 0.5, 0.75]}),
    ('GridDistortion',    {'num_steps': (3, 5), 'distort_limit': (
        0.1, 0.5),  'p': [0.0, 0.25, 0.5, 0.75]}),
    ('ElasticTransform',  {'alpha': 1, 'sigma': (30, 70),
                           'alpha_affine': (30, 70),  'p': [0.0, 0.25, 0.5, 0.75]}),
]


def sample_params(r):
    if isinstance(r, list):
        return random.choice(r)

    if not isinstance(r, tuple):
        return r

    r1, r2 = r
    if not isinstance(r1, tuple):
        assert not isinstance(r2, tuple)
        if isinstance(r1, float):
            return random.uniform(r1, r2)
        else:
            return random.randint(r1, r2)

    assert isinstance(r1, tuple)
    assert isinstance(r2, tuple)
    return (sample_params(r1), sample_params(r2))


def sample_policy(num_sub_policy, op_per_sub):
    policies = []

    for sub_policy_idx in range(num_sub_policy):
        op_0, op_1 = random.sample(op_options, op_per_sub)
        params_0 = {key: sample_params(value)
                    for key, value in op_0[1].items()}
        params_1 = {key: sample_params(value)
                    for key, value in op_1[1].items()}
        sub_policy = (
            (op_0[0], params_0),
            (op_1[0], params_1)
        )
        policies.append(sub_policy)
    return policies
