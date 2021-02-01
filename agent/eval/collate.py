from rlenv.generate.util import process_sims
from agent.eval.util import sim_args
from agent.util import get_output_dir
from utils import unpickle
from constants import NUM_CHUNKS


def main():
    args = sim_args()

    # output directory
    output_dir = get_output_dir(part=args.part,
                                heuristic=args.heuristic,
                                byr=args.byr,
                                delta=args.delta,
                                turn_cost=args.turn_cost)

    # concatenate
    sims = []
    for i in range(NUM_CHUNKS):
        chunk_path = output_dir + 'outcomes/{}.pkl'.format(i)
        sims.append(unpickle(chunk_path))

    # clean and save
    process_sims(part=args.part, sims=sims, output_dir=output_dir)


if __name__ == '__main__':
    main()
