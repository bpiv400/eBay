import sys
from compress_pickle import dump
import pandas as pd, numpy as np
from constants import *

OTYPES = {'lstg': 'int64',
		  'thread': 'int64',
		  'index': 'uint8',
		  'clock': 'int64', 
		  'price': 'float64', 
		  'accept': bool,
		  'reject': bool,
		  'censored': bool,
		  'message': bool}

TTYPES = {'lstg': 'int64',
		  'thread': 'int64',
		  'byr': 'int64',
		  'byr_hist': 'int64',
		  'start_time': 'int64',
		  'bin': bool,
		  'byr_us': bool}

LTYPES = {'lstg': 'int64',
		  'slr': 'int64',
		  'meta': 'uint8',
		  'cat': str,
		  'cndtn': 'uint8',
		  'start_date': 'uint16',
		  'end_time': 'int64',
		  'fdbk_score': 'int64',
		  'fdbk_pstv': 'int64',
		  'start_price': 'float64',
		  'photos': 'uint8',
		  'slr_lstgs': 'int64',
		  'slr_bos': 'int64',
		  'decline_price': 'float64',
		  'accept_price': 'float64',
		  'store': bool,
		  'slr_us': bool,
		  'flag': bool,
		  'toDrop': bool,
		  'fast': bool}
		  

# offers
O = pd.read_csv(CLEAN_DIR + 'offers.csv', dtype=OTYPES).set_index(
	['lstg', 'thread', 'index'])
dump(O, CLEAN_DIR + 'offers.pkl')
del O

# threads
T = pd.read_csv(CLEAN_DIR + 'threads.csv', dtype=TTYPES).set_index(
	['lstg', 'thread'])
dump(T, CLEAN_DIR + 'threads.pkl')
del T

# listings
L = pd.read_csv(CLEAN_DIR + 'listings.csv', dtype=LTYPES).set_index(
	'lstg')
dump(L, CLEAN_DIR + 'listings.pkl')