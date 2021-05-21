import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = True


def get_name(path):
    return path.split('/')[-1].split('_')[1]


def add_diagonal(df):
    low, high = df.index.min(), df.index.max()
    plt.plot([low, high], [low, high], '-k', lw=0.5)
