from discrim.DiscrimGenerator import DiscrimGenerator
from rlenv.generate.util import process_sims
from utils import run_func_on_chunks, process_chunk_worker, input_partition
from constants import NUM_RL_WORKERS, PARTS_DIR


def main():
    part = input_partition()

    # process chunks in parallel
    print('Generating {} discriminator input'.format(part))
    sims = run_func_on_chunks(
        num_chunks=NUM_RL_WORKERS,
        f=process_chunk_worker,
        func_kwargs=dict(
            part=part,
            gen_class=DiscrimGenerator,
            gen_kwargs=dict(verbose=False)
        )
    )

    # concatenate, clean, and save
    process_sims(part=part, sims=sims, parent_dir=PARTS_DIR)


if __name__ == '__main__':
    main()
