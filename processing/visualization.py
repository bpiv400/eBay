import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe


def main():
    data = pd.read_csv('data/train.csv')
    offr = data['offr_price'].value
    hist = sns.distplot(offr, kde=False)
    hist.savefig('data/hist.png')
    box = sns.boxplot(x=offr)
    box.savefig('data/box.png')
    print(describe(offr))


if __name__ == '__main__':
    main()
