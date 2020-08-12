import argparse
import torch
from agent.AgentModel import AgentModel
from agent.eval.EvalGenerator import EvalGenerator
from agent.util import get_paths
from rlenv.environments.SellerEnv import SellerEnv
from rlenv.environments.BuyerEnv import BuyerEnv
from rlenv.generate.util import process_sims
from utils import set_gpu_workers, run_func_on_chunks, process_chunk_worker
from agent.const import AGENT_STATE, OPTIM_STATE
from constants import BYR, VALIDATION, NUM_RL_WORKERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--entropy_coeff', type=float, default=.001)
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    args = vars(parser.parse_args())

    # set gpu and cpu affinity
    set_gpu_workers(gpu=args['gpu'], spawn=True)

    # environment class and run directory
    env = BuyerEnv if args[BYR] else SellerEnv
    _, _, run_dir = get_paths(**args)

    # recreate model
    model = AgentModel(byr=args[BYR])
    path = run_dir + 'params.pkl'
    d = torch.load(path, map_location=torch.device('cpu'))
    if OPTIM_STATE in d:
        d = d[AGENT_STATE]
        torch.save(d, path)
    model.load_state_dict(d, strict=True)

    # run in parallel on chunks
    sims = run_func_on_chunks(
        num_chunks=NUM_RL_WORKERS,
        f=process_chunk_worker,
        func_kwargs=dict(
            part=VALIDATION,
            gen_class=EvalGenerator,
            gen_kwargs=dict(env=env, model=model)
        )
    )

    # combine and process output
    part_dir = run_dir + '{}/'.format(VALIDATION)
    process_sims(part=VALIDATION, sims=sims, output_dir=part_dir)


if __name__ == '__main__':
    main()
