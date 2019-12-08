import argparse
import h5py


COLS = ['taco', 'dog', 'man with a hat']
PATH = 'test.hdf5'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode', action='store_true')
    decode = parser.parse_args().decode
    print('Decode: {}'.format(decode))
    if decode:
        decoder()
    else:
        encoder()


def decoder():
    h5file = h5py.File(PATH, 'r')
    dset = h5file['lookup']
    cols = dset.attrs['cols']
    cols = [col.decode('utf-8') for col in cols]
    col_eq = [fixed_col == decode_col for fixed_col, decode_col in zip(COLS, cols)]
    for col in col_eq:
        assert col


def encoder():
    h5file = h5py.File(PATH, 'w')
    encoded_cols = [col.encode('utf-8') for col in COLS]
    dset = h5file.create_dataset('lookup', dtype="f")
    dset.attrs['cols'] = encoded_cols
    h5file.close()


if __name__ == '__main__':
    main()