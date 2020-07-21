import os
import pandas as pd
from constants import POLICY_SLR, POLICY_BYR, DROPOUT
from agent.const import AGENT_PARAMS, SYSTEM_PARAMS


def save_run(log_dir=None, run_id=None, args=None):
    run_path = log_dir + 'runs.csv'
    if os.path.isfile(run_path):
        df = pd.read_csv(run_path, index_col=0)
    else:
        df = pd.DataFrame(index=pd.Index([], name='run_id'))

    exclude = list({**AGENT_PARAMS, **SYSTEM_PARAMS}.keys())
    exclude += [DROPOUT]
    exclude.remove('batch_size')
    for k, v in args.items():
        if k not in exclude:
            df.loc[run_id, k] = v
    df.to_csv(run_path)


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR
