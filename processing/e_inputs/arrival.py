from compress_pickle import load
from processing.processing_utils import input_partition, load_file, \
    process_arrival_inputs, save_files
from processing.processing_consts import CLEAN_DIR


def main():
    # partition name from command line
    part = input_partition()
    print('%s/arrival' % part)

    # load timestamps
    lstg_start = load_file(part, 'lookup').start_time
    lstg_end = load(CLEAN_DIR + 'listings.pkl').end_time.reindex(
        index=lstg_start.index)
    thread_start = load_file(part, 'clock').xs(1, level='index')

    # input dataframes, output processed dataframes
    d = process_arrival_inputs(part, lstg_start, lstg_end, thread_start)

    # save various output files
    save_files(d, part, 'arrival')
    

if __name__ == '__main__':
    main()