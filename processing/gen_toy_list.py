import numpy as np
import pandas as pd

lists = pd.read_csv('data/lists.csv')
lists.set_index('anon_item_id', inplace=True)
toy = pd.read_csv('data/toy.csv')

# get item id's
ids = toy['anon_item_id'].values
toy_lists = lists.loc[ids]
toy_lists.to_csv('data/toy_lists.csv')
