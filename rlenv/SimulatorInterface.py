# TODO UPDATE DOCUMENTATION
class SimulatorInterface:
    def __init__(self, params):
        """
        Use rl experiment params to initialize models

        :param params:
        """
        pass

    def arrival_time(self, consts=None, time_feats=None, hidden=None):
        """
        given a set of constants about a lstg
        simulate an arrival time

        Args:
            consts: np.numpy array
            time_feats: integer giving the current time
            hidden: hidden state from previous step of arrival process for this lstg. None if this is the first step
            This should be a torch.double
        Return:
            Tuple of whether a buyer arrives in the next interval and a hidden state output
        """
        raise NotImplementedError()

    def buyer_delay(self, consts=None, time_feats=None, delay=None,
                    prev_slr_offer=None, prev_byr_offer=None, hidden=None):
        """
        Given a pair of previous offers, the hidden state for the delay model, a set of constant features for
        the listing, the length of the delay thus far, and a set of time features, generate a realization for whether
        the seller makes an offer in the next interval

        Additionally update hidden state as necessary

        Args:
            hidden: torch.tensor giving the previous hidden state for the delay model
            consts: 1 dimensional np.array of constant features for the lstg
            time_feats: output of TimeFeatures.get_features (currently a dictionary) containing all time valued features
            prev_slr_offer: offer representation of last seller offer output by
            SimulatorInterface.seller_offer (currently a dictionary). None if no seller offer has occurred
            prev_byr_offer: offer representation of last buyer offer output by
            SimulatorInterface.buyer_offer (currently a dictionary)
            delay: integer giving the delay associated with this offer
        Returns:
            Tuple of an offer realization (1 if an offer is made in the next interval, 0 otherwise)
            and updated hidden state after this step. None if an offer is made
        """
        raise NotImplementedError()

    def seller_delay(self, consts=None, time_feats=None, delay=None,
                     prev_slr_offer=None, prev_byr_offer=None, hidden=None):
        """
        Given a pair of previous offers, the hidden state for the delay model, a set of constant features for
        the listing, the length of the delay thus far, and a set of time features, generate a realization for whether
        the seller makes an offer in the next interval

        Additionally update hidden state as necessary

        Args:
            hidden: torch.tensor giving the previous hidden state for the delay model
            consts: 1 dimensional np.array of constant features for the lstg
            time_feats: output of TimeFeatures.get_features (currently a dictionary) containing all time valued features
            prev_slr_offer: offer representation of last seller offer output by
            SimulatorInterface.seller_offer (currently a dictionary). None if no seller offer has occurred
            prev_byr_offer: offer representation of last buyer offer output by
            SimulatorInterface.buyer_offer (currently a dictionary)
            delay: integer giving the delay associated with this offer
        Returns:
            Tuple of an offer realization (1 if an offer is made in the next interval, 0 otherwise)
            and updated hidden state after this step. None if an offer is made
        :return:
        """
        raise NotImplementedError()

    def buyer_offer(self, consts=None, hidden=None, time_feats=None, prev_slr_offer=None, prev_byr_offer=None,
                    prev_slr_delay=None, prev_byr_delay=None, delay=None):
        """
        Given a previous buyer offer, a previous seller offer, a dictionary of previous hidden states,
        a set of time valued features, and the amount of delay chosen for this offer, simulate the next buyer offer

        Also updates the relevant hidden states
        Args:
            hidden: dictionary of tensors giving previous hidden states
            of the buyer simulator models ('concession', 'delay', 'round')
            consts: 1 dimensional np.array of constant features for the lstg
            time_feats: output of TimeFeatures.get_features (currently a dictionary) containing all time valued features
            prev_slr_offer: offer representation of last seller offer output by
            SimulatorInterface.seller_offer (currently a dictionary). None if no seller offer has occurred
            prev_byr_offer: offer representation of last buyer offer output by
            SimulatorInterface.buyer_offer (currently a dictionary)
            prev_slr_delay: integer giving the number of seconds the seller delayed for their previous offer
            prev_byr_delay: integer giving the number of seconds the buyer delayed for their previous offer
            delay: integer giving the delay associated with this offer
        Returns:
            Tuple of an offer representation (currently a dictionary that is expected to contain concession [0, 1] and
            price (real price of the offer)--nothing else is assumed. Should contain all other features used
            as inputs later AND hidden state dictionary (should have delay = None, and concession/round from this
            most recent step
        """
        raise NotImplementedError()

    def slr_offer(self, consts=None, hidden=None, time_feats=None, prev_slr_offer=None, prev_byr_offer=None,
                  prev_slr_delay=None, prev_byr_delay=None, delay=None):
        """
        Given a previous buyer offer, a previous seller offer, a dicitonary of previous hidden states for the
        seller models, a set of time valued features,
        and the amount of delay chosen for this offer, simulate the next seller offer

        Also updates the relevant hidden states
        Args:
            hidden: dictionary of tensors giving previous hidden states
            of the buyer simulator models ('concession', 'delay', 'round')
            consts: 1 dimensional np.array of constant features for the lstg
            time_feats: output of TimeFeatures.get_features (currently a dictionary) containing all time valued features
            prev_slr_offer: offer representation of last seller offer output by
            SimulatorInterface.seller_offer (currently a dictionary). None if no seller offer has occurred
            prev_byr_offer: offer representation of last buyer offer output by
            SimulatorInterface.buyer_offer (currently a dictionary)
            prev_slr_delay: integer giving the number of seconds the seller delayed for their previous offer
            prev_byr_delay: integer giving the number of seconds the buyer delayed for their previous offer
            delay: integer giving the delay associated with this offer
        Returns:
            Tuple of an offer representation (currently a dictionary that is expected to contain concession [0, 1] and
            price (real price of the offer)--nothing else is assumed. Should contain all other features used
            as inputs later AND hidden state dictionary (should have delay = None, and concession/round from this
            most recent step
            """
        raise NotImplementedError()
