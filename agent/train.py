import argparse
import os
import psutil
import torch
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.utils.logging.context import logger_context
from agent.AgentComposer import AgentComposer
from agent.AgentLoader import AgentLoader
from agent.AgentModel import AgentModel
from agent.EBayRunner import EBayRunner
from agent.EBayPPO import EBayPPO
from agent.agents.SellerAgent import SellerAgent
from agent.agents.BuyerAgent import BuyerAgent
from agent.envs.SellerEnv import SellerEnv
from agent.envs.BuyerEnv import BuyerEnv
from agent.util import get_run_dir, get_log_dir, get_run_id
from env.QueryStrategy import DefaultQueryStrategy
from env.Player import SimulatedSeller, SimulatedBuyer
from utils import set_gpu
from agent.const import DELTA_BYR, DELTA_SLR, TURN_COST_CHOICES, BATCH_SIZE
from featnames import DELTA, TURN_COST


def main():
    # command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--delta', type=float, required=True)
    parser.add_argument('--turn_cost', type=int, default=0,
                        choices=TURN_COST_CHOICES)
    args = parser.parse_args()
    serial = args.gpu is None

    # error checking
    if args.log:
        assert not serial

    if args.byr:
        assert args.delta in DELTA_BYR
    else:
        assert args.delta in DELTA_SLR

    # make sure run doesn't already exist
    if args.log:
        run_dir = get_run_dir(byr=args.byr,
                              delta=args.delta,
                              turn_cost=args.turn_cost)
        if os.path.isdir(run_dir):
            print('Run already exists.')
            exit()

    # set gpu
    if not serial:
        set_gpu(gpu=args.gpu)

    # print to console
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))

    # create sampler
    if serial:
        sampler_cls = SerialSampler
        batch_b, batch_t = 1, 64
        affinity = None
    else:
        sampler_cls = AlternatingSampler
        cpus = list(psutil.Process().cpu_affinity())
        # workers = [w for w in cpus if w % 2 == (args.gpu % 2)]
        workers = cpus
        affinity = dict(workers_cpus=workers + workers,
                        cuda_idx=torch.cuda.current_device(),
                        set_affinity=True,
                        alternating=True)

        batch_b = 2 * len(workers)
        batch_t = int(BATCH_SIZE / batch_b)

    qs = DefaultQueryStrategy(
        seller=SimulatedSeller(full=args.byr),
        buyer=SimulatedBuyer(full=True)
    )
    env_kwargs = dict(
        composer=AgentComposer(byr=args.byr),
        verbose=serial,
        query_strategy=qs,
        loader=AgentLoader(num_workers=batch_b),
        delta=args.delta,
        turn_cost=args.turn_cost,
        train=True
    )
    sampler = sampler_cls(
        batch_B=batch_b,
        batch_T=batch_t,
        EnvCls=BuyerEnv if args.byr else SellerEnv,
        env_kwargs=env_kwargs,
        max_decorrelation_steps=0,)

    # construct agent
    model_kwargs = dict(byr=args.byr, turn_cost=args.turn_cost)
    agent_kwargs = dict(ModelCls=AgentModel,
                        model_kwargs=model_kwargs,
                        serial=serial)
    if args.byr:
        agent_kwargs[DELTA] = args.delta
        agent_kwargs[TURN_COST] = args.turn_cost
    agent_cls = BuyerAgent if args.byr else SellerAgent
    agent = agent_cls(**agent_kwargs)

    # create runner
    runner = EBayRunner(algo=EBayPPO(),
                        agent=agent,
                        sampler=sampler,
                        affinity=affinity)

    # train
    if not args.log:
        runner.train()
    else:
        with logger_context(log_dir=get_log_dir(byr=args.byr),
                            name='log',
                            use_summary_writer=True,
                            override_prefix=True,
                            run_ID=get_run_id(byr=args.byr,
                                              delta=args.delta,
                                              turn_cost=args.turn_cost),
                            snapshot_mode='last'):
            runner.train()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
