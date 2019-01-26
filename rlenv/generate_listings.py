"""
generate_listings.py

Interface for creating listing objects from raw data files
Should be processed as set of array jobs
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
                        type=bool, required=False, default=False)
    # parse args
    args = parser.parse_args()
    # define environment path
    path = 'data/datasets/%s/listings/env.pkl' % args.data
    # if this is the first listing chunk
    if not args.init:
        env = ListingEnvironment.load(args.data)
        env.generate_data(args.data, args.name, new_env=False)
    else:
        # otherwise create the environment
        env = ListingEnvironment(
            data_name=args.data, chunk=args.name, generate_data=True)
        env.save()


if __name__ == '__main__':
    main()
