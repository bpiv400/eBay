"""
Adds features to listings dataframe:
-quality indicator
-sale time, if not included
-metacategory id indicator
"""

# modules
import argparse
import pandas as pd
import numpy as np


def main():
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', action='store', required=True)
    chunk_name = parser.parse_args().name
    datatype = extract_datatype(chunk_name)
    # load lstings
    lstg_path = 'data/%s/offers/%s_offers.csv' % (datatype, chunk_name)
    lstgs = pd.read_pickle(lstg_path)
    lstgs =


if __name__ == "__main__":
    main()
