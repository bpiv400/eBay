import numpy as np
import torch
from agent.models.util import wrapper, get_store, get_last_norm, \
    get_last_auto, get_agent_turn, get_start_price
from agent.const import DELTA_BYR
from constants import NUM_COMMON_CONS
from featnames import DELTA, TURN_COST


class HeuristicByr:
    def __init__(self, **params):
        self.high = np.isclose(params[DELTA], DELTA_BYR[-1])
        self.turn_cost = params[TURN_COST]

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        x = observation._asdict()

        # turn number
        turn = get_agent_turn(x=x, byr=True)

        # index of action
        f = wrapper(turn)
        if turn == 1:
            if not self.high:
                store = get_store(x)
                idx = f(0) if store else f(.5)
            elif self.turn_cost == 0:
                idx = f(.5)
            elif self.turn_cost == 1:
                idx = f(.5)
            else:
                start_price = get_start_price(x)
                if self.turn_cost == 3:
                    idx = f(0) if start_price <= 45 else f(.5)
                elif self.turn_cost == 4:
                    idx = f(0) if start_price <= 60 else f(.5)
                elif self.turn_cost == 5:
                    idx = f(0) if start_price < 72 else f(.5)
                else:
                    raise RuntimeError("")

        elif turn == 3:
            if not self.high:
                idx = f(.17)
            elif self.turn_cost == 0:
                auto = get_last_auto(x=x, turn=turn)
                idx = f(.4) if auto else f(.17)
            elif self.turn_cost == 1:
                idx = f(.4)
            elif self.turn_cost == 3:
                auto = get_last_auto(x=x, turn=turn)
                idx = f(.4) if auto else f(0)
            elif self.turn_cost == 4:
                auto = get_last_auto(x=x, turn=turn)
                idx = f(0) if auto else f(.25)
            elif self.turn_cost == 5:
                norm = get_last_norm(turn=turn, x=x)
                idx = f(0) if norm >= .855 else f(.4)
            else:
                raise RuntimeError("")

        elif turn == 5:
            if not self.high:
                idx = f(.17)
            elif self.turn_cost == 0:
                idx = f(.4)
            elif self.turn_cost == 1:
                idx = f(.4)
            elif self.turn_cost == 3:
                auto = get_last_auto(x=x, turn=turn)
                idx = f(.4) if auto else f(.17)
            elif self.turn_cost == 4:
                idx = f(.25)
            elif self.turn_cost == 5:
                norm = get_last_norm(turn=turn, x=x)
                idx = f(.4) if norm > .79 else f(1)
            else:
                raise RuntimeError("")

        elif turn == 7:
            if not self.high:
                idx = f(0)
            else:
                norm = get_last_norm(turn=turn, x=x)
                if self.turn_cost == 0:
                    tau = .91
                elif self.turn_cost == 1:
                    tau = .91
                elif self.turn_cost == 3:
                    tau = .91
                elif self.turn_cost == 4:
                    tau = .83
                elif self.turn_cost == 5:
                    tau = .86
                else:
                    raise RuntimeError("")
                idx = f(1) if norm <= tau else f(0)
        else:
            raise ValueError('Invalid turn: {}'.format(turn))

        # deterministic categorical action distribution
        pdf = torch.zeros(NUM_COMMON_CONS + 2, dtype=torch.float)
        pdf[idx] = 1.
        return pdf
