"""
Script for updating the indices files that Composer uses to
populate the model input tensors
"""

from rlenv.composer.Composer import Composer


def main():
    """
    Main method
    :return: NA
    """
    Composer(rebuild=True)


if __name__ == '__main__':
    main()
