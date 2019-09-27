from model_names import *


def model_str(model_name, byr=False):
    """

    :param model_name:
    :param slr:
    :return:
    """
    if not byr:
        name = '{}_{}'.format(SLR_PREFIX, model_name)
    else:
        name = '{}_{}'.format(BYR_PREFIX, model_name)
    return name
