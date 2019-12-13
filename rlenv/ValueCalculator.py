from rlenv.env_utils import get_cut
from rlenv.env_consts import META

class ValueCalculator:
    def __init__(self):
        self.sale_sum = 0
        self.sale_count = 0
        self.cut = 0

    def update_lstg(self, lookup):
        self.sale_count = 0
        self.cut = get_cut(lookup[META])