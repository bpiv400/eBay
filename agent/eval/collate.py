from shutil import rmtree
from env.generate.util import process_sims
from agent.eval.util import sim_args
from agent.util import get_sim_dir
from utils import unpickle
from constants import NUM_CHUNKS


def main():
    args = sim_args()

    # output directory
    sim_dir = get_sim_dir(**vars(args))
    outcome_dir = sim_dir + 'outcomes/'

    # concatenate
    sims = []
    for i in range(NUM_CHUNKS):
        chunk_path = outcome_dir + '{}.pkl'.format(i)
        sims.append(unpickle(chunk_path))

    # clean and save
    process_sims(part=args.part, sims=sims, sim_dir=sim_dir)

    # delete chunk files
    rmtree(outcome_dir)


if __name__ == '__main__':
    main()
