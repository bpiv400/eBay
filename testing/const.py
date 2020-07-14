from constants import BYR, SLR, VALIDATION, PARTITIONS

SCRIPT_PARAMS = {'role': {'type': str,
                          'choices': [BYR, SLR],
                          'default': SLR},
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

TEST_GENERATOR_KWARGS = ['start', 'role', 'agent', 'verbose', 'part']
