from constants import BYR, VALIDATION, PARTITIONS

SCRIPT_PARAMS = {BYR: {'action': 'store_true'},
                 'agent': {'action': 'store_true'},
                 'part': {'type': str,
                          'default': VALIDATION,
                          'choices': PARTITIONS
                          },
                 'num': {'type': int,
                         'default': 1},
                 'start': {'type': int,
                           'required': False},
                 'verbose': {'action': 'store_true'}
                 }

TEST_GENERATOR_KWARGS = ['start', BYR, 'agent', 'verbose', 'part']
