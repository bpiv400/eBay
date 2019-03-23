import pickle
import pandas as pd, numpy as np
from gensim import models as lda

DIR = './data/lda/'

if __name__ == "__main__":
    # number of topics
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics', action='store', type=int, required=True)
    topics = parser.parse_args().topics

    # input
    tfidf = pickle.load(open(DIR + 'tfidf.pkl', 'rb'))

    # train model
    print('Training model')
    model = lda.LdaModel(tfidf, num_topics=topics, eta='auto')

    # save model
	pickle.dump(model, open(DIR + 'm' + str(topics) + '.pkl', 'wb'))
