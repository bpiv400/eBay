import torch
import shap
from sim.EBayDataset import EBayDataset
from sim.Sample import get_batches
from agent.models.AgentModel import load_agent_model
from agent.util import get_log_dir
from agent.const import FULL
from constants import TEST, POLICY_SLR

torch.multiprocessing.set_start_method('fork')

# run folder
slr_dir = get_log_dir(False) + 'run_full_fewer_feats_smaller_model/'

# create model
model_args = dict(byr=False, con_set=FULL, value=False)
model = load_agent_model(model_args=model_args,
                         run_dir=slr_dir).to('cuda')

# create dataset
data = EBayDataset(part=TEST, name=POLICY_SLR, folder=slr_dir)
batches = get_batches(data, is_training=False)
for b in batches:
    print(b)
    break

# kernel explainer
explainer = shap.KernelExplainer(model, data)
