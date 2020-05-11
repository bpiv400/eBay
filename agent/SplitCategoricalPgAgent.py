from rlpyt.agents.pg.categorical import CategoricalPgAgent


class SplitCategoricalPgAgent(CategoricalPgAgent):
    def value_parameters(self):
        return self.model.value_parameters()

    def policy_parameters(self):
        return self.model.policy_parameters()

    def zero_values_grad(self):
        self.model.zero_values_grad()
