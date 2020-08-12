from rlpyt.agents.pg.categorical import CategoricalPgAgent


class SplitCategoricalPgAgent(CategoricalPgAgent):
    def value_parameters(self):
        return self.model.value_net.parameters()

    def policy_parameters(self):
        return self.model.policy_net.parameters()