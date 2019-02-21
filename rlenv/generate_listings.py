"""
generate_listings.py

Interface for creating listing objects from raw data files

Associated scripts: gen_init_listing.sh, gen_late_listings.sh, gen_listings.sh
"""
import os
import argparse
from listings import ListingEnvironment


def main():
    """
    Generates listing binaries for a specific dataset and data chunk
    """
    parser = argparse.ArgumentParser()
    # gives the name of the type of group we're operating on
    # toy, train, test, pure_test (not the literal filename)
    parser.add_argument('--name', '-n', action='store',
                        type=str, required=True)
    # gives the name of the current data type
    parser.add_argument('--data', '-d', action='store',
                        type=str, required=True)
    # first listing chunk
    parser.add_argument('--init', '-i', action='store_true',
                        required=False, default=False)
    # parse args
    args = parser.parse_args()
    # define environment path
    path = 'data/datasets/%s/listings/env.pkl' % args.data
    # if this is the first listing chunk
    if not args.init:
        env = ListingEnvironment.load(data_name=args.data)
        env.gen_data(args.name, new_env=False)
    else:
        # otherwise create the environment
        env = ListingEnvironment(data_name=args.data, chunk=args.name)
        env.save()


if __name__ == '__main__':
    main()
