from matplotlib import pyplot as plt
from agent.const import DELTA_BYR
from plots.save import save_fig
from plots.util import get_name


def plot_humans(df=None, offsets=None):
    humans = df.loc['Humans', :]
    plt.plot(humans.x, humans.y, 'Xk')
    plt.text(humans.x + offsets[0], humans.y + offsets[1], 'Humans')


def plot_agents(df=None, offsets=None, name=None, lspec='-o', color='k'):
    labels = []
    for d in DELTA_BYR:
        if d != 1:
            label = '${}$'.format(d)
            if label in df.index:
                labels.append(label)
        else:
            labels.append('$1-\\epsilon$')
            labels.append('$1+\\epsilon$')
    agents = df.loc[labels, :]
    plt.plot(agents.x, agents.y, lspec, label=name, color=color)
    if offsets is not None:
        for label in labels:
            if label in agents.index:
                plt.text(agents.loc[label, 'x'] + offsets[0],
                         agents.loc[label, 'y'] + offsets[1],
                         label)


def pareto_plot(path, d):
    name = get_name(path)
    if name == 'discount':
        args = dict(xlabel='Purchase rate',
                    ylabel='Discount on list price (%)',
                    ylim=[15, 45], xlim=[.55, 1],
                    legend=False)
    elif name == 'dollar':
        args = dict(xlabel='Purchase rate',
                    ylabel='Discount on list price ($)',
                    ylim=[15, 45], xlim=[.55, 1],
                    legend=True)
    elif name == 'sales':
        args = dict(xlabel='Purchase rate',
                    ylabel='Savings on observed sale price ($)',
                    ylim=[0, 20], xlim=[.55, 1])
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    offsets = [(args[k][1] - args[k][0]) / 100 for k in ['xlim', 'ylim']]

    plot_humans(df=d['full'], offsets=offsets)
    plot_agents(df=d['full'], offsets=offsets, name='Agents')
    plot_agents(df=d['heuristic'], name='Heuristics', color='gray')
    save_fig(path, **args)
