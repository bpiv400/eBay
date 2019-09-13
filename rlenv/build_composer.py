"""
Script for updating the indices files that Composer uses to
populate the model input tensors
"""

from rlenv.Composer import Composer
def main():
    """
    Main method
    :return: NA
    """
    composer = Composer()
    composer.build()


if __name__ == '__main__':
    main()
