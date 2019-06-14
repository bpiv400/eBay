"""
File containing deprecated functions and classes
from rlenv portion of project
"""


@staticmethod
def get_critic_class(exp_name):
    '''
    Function to parse experiment name for class of
    critic model...For now,
    trivially returns Critic
    '''
    return Critic


def get_next_offr(self, recent_offr):
    _, next_state = self.lstm(recent_offr, (self.h1, self.c1))
    self.h1, self.c1 = next_state

    return self.action_from_state(self.h1)


@staticmethod
def get_actor_class(exp_name):
    '''
    Function to parse experiment name for class
    of actor model...For now,
    trivially returns DeterministicActor
    '''
    if exp_name is None:
        raise ValueError("Must pass experiment name")
    return DeterministicActor


def __get_params(self, name):
    '''
    Parses the name of the experiment and the data stored
    from __extract_data(...) to determine the parameters
    of the transition probability model

    Called after __extract_data(...)

    Args:
        None
    Returns:
        Dictionary string -> val
    '''
    params = {}
    params['num_offr_feats'] = self.offrs.shape[2]
    params['org_hidden_size'] = self.consts.shape[2]
    params['init'] = Simulator.get_init(name)
    params['lstm'] = Simulator.get_lstm(name)
    params['zeros'] = Simulator.get_zeros(name)
    params['num_layers'] = Simulator.get_num_layers(name)
    params['targ_hidden_size'] = Simulator.get_hidden_size(self.consts,
                                                           name)
    params['bet_hidden_size'] = Simulator.get_bet_size(
        name)
    return params


def __extract_data(self):
    '''
    DEPRECATED
    TODO: REMOVE

    Private helper method that parses the experiment name
    to extract the data type then loads the corresponding
    data and feature dictionaries and stores their items
    in local variables

    Args:
        None
    Return:
        None
    '''
    # parse name of data from experiment name
    data_name = get_data_name(self.trans_name)
    # assign names of feature and data dictionaries
    data_loc = 'data/exps/%s/train_data.pickle' % data_name
    feats_loc = 'data/exps/%s/feats.pickle' % data_name
    self.feats_dict = unpickle(feats_loc)
    data_dict = unpickle(data_loc)

    # extract necessary components from data_dict
    # num_state_layers x n x state_size
    self.consts = data_dict['const_vals']
    # max_offers x n x feats_per_offer
    self.offrs = data_dict['offr_vals']
    # max_offers x n
    self.targs = data_dict['target_vals']
    # series where the values give the midpoint of each
    # offer bin and the indices give their corresponding
    # class indices from the perspective of the
    # trans probs model
    self.class_to_action_ser = data_dict['midpoint_ser']

    # series where the values give the offer class index
    # and the index gives the value of the offer
    self.action_to_class_ser = pd.Series(self.class_to_action_ser.index,
                                         index=np.round(self.class_to_action.values, 2))
    # n x 1 numpy vector of lengths
    self.lengths = data_dict['length_vals']


def get_data_name(exp_name):
    '''
    DEPRECATED
    See apinotes.txt for handling of parameters

    Parse name of the dataset being used out of the
    name of the transition probability experiment used

    Args:
        exp_name: string giving name of transition probability
        experiment
    '''

    if 'lstm' in exp_name:
        data_name = exp_name.replace('lstm', 'rnn')
    else:
        data_name = exp_name
    arch_type_str = r'_(simp|cat|sep)'
    type_match = re.search(arch_type_str, data_name)
    if type_match is None:
        raise ValueError('Invalid experiment name')
    type_match_end = type_match.span(0)[1]
    data_name = data_name[:type_match_end]
    return data_name
