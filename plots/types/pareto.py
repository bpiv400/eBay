from matplotlib import pyplot as plt
from plots.save import save_fig
from plots.util import get_name


def plot_humans(df=None, offsets=None):
    humans = df.loc['Humans', :]
    plt.plot(humans.x, humans.y, 'Xk')
    plt.text(humans.x + offsets[0], humans.y + offsets[1], 'Humans')


def add_labels(df=None, labels=None, offsets=None):
    if offsets is not None:
        for label in labels:
            if label in df.index:
                plt.text(df.loc[label, 'x'] + offsets[0],
                         df.loc[label, 'y'] + offsets[1],
                         label)


def plot_agents(df=None, offsets=None, color='k', labels=None):
    agents = df.drop('Humans')
    plt.plot(agents.x, agents.y, '-o', label='Agents', color=color)
    if labels is None:
        labels = agents.index
    add_labels(df=agents, labels=labels, offsets=offsets)


def plot_heuristics(df=None):
    plt.plot(df.x, df.y, '-o', label='Heuristics', color='gray')


def plot_turn_cost(d=None, offsets=None):
    for k, df in d.items():
        if k in ['plus', 'minus']:
            plt.plot(df.x, df.y, '-o', color='k')
            add_labels(df=df, labels=df.index, offsets=offsets)


def pareto_plot(path, d):
    name = get_name(path)

    if name == 'best':
        yticks = list(range(-2, 12, 2))
        args = dict(xlabel='Purchase rate',
                    ylabel='Savings on best human buyer offer',
                    ylim=[-2, 10], xlim=[.65, 1],
                    yticks=yticks,
                    yticklabels=['${}'.format(t)
                                 if t >= 0 else '-${}'.format(-t)
                                 for t in yticks],
                    legend=False)
    else:
        args = dict(xlabel='Purchase rate',
                    ylabel='Discount on list price',
                    ylim=[15, 40], xlim=[.65, 1],
                    yticks=range(15, 45, 5),
                    legend=(name == 'discount'))
        if name in ['discount', 'cost']:
            args['yticklabels'] = ['{}%'.format(t) for t in args['yticks']]
        elif name == 'dollar':
            args['yticklabels'] = ['${}'.format(t) for t in args['yticks']]
        else:
            raise NotImplementedError('Invalid name: {}'.format(name))

    offsets = [(args[k][1] - args[k][0]) / 100 for k in ['xlim', 'ylim']]

    plot_humans(df=d['full'], offsets=offsets)
    if name == 'cost':
        plot_agents(df=d['full'], offsets=offsets, color='gray',
                    labels=['$1+\\epsilon$'])
        plot_turn_cost(d, offsets=[-4 * off for off in offsets])
    else:
        plot_agents(df=d['full'], offsets=offsets)
        plot_heuristics(df=d['heuristic'])
    save_fig(path, **args)
