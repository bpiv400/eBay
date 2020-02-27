from shutil import copyfile
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
import numpy as np
from processing.processing_consts import LOG_DIR
from constants import MODEL_DIR, MODELS


for model in MODELS:
	em = EventMultiplexer().AddRunsFromDirectory(LOG_DIR + model)
	em.Reload()

	lnL_test = dict()
	for run, d in em.Runs().items():
		if len(d['scalars']) > 0:
			res = em.Scalars(run, 'lnL_test')
			if len(res) > 6:  # restrict to runs with more than 6 epochs
				lnL_test[run] = em.Scalars(run, 'lnL_test')[-1].value

	idx = np.argmax(list(lnL_test.values()))
	best = list(lnL_test.keys())[idx]

	# copy best performing model into parent directory
	print('{}: {}'.format(model, best))
	copyfile(MODEL_DIR + '{}/{}.net'.format(model, best),
			 MODEL_DIR + '{}.net'.format(model))