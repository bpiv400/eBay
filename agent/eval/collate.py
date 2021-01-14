from rlenv.generate.util import process_sims
from agent.eval.util import get_output_dir, sim_args
from utils import unpickle
from constants import NUM_CHUNKS
from featnames import BYR


def main():
    args = sim_args()

    # output directory
    output_dir = get_output_dir(**args)

    # concatenate
    sims = []
    for i in range(NUM_CHUNKS):
        chunk_path = output_dir + 'outcomes/{}.pkl'.format(i)
        sims.append(unpickle(chunk_path))

    # clean and save
    process_sims(part=args['part'],
                 sims=sims,
                 output_dir=output_dir,
                 byr=args[BYR],
                 save_inputs=(not args['heuristic']))


if __name__ == '__main__':
    main()
