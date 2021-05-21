from agent.const import DELTA_SLR
from plots.const import ALTERNATING
from plots.save import save_fig
from plots.util import get_name


def area_plot(path, df):
    name = get_name(path)
    if name == 'turncon':
        args = dict(xlim=[1, 7], ylim=[.6, 1],
                    xlabel='Day of first offer',
                    ylabel='Concession / list price')
        if path.endswith(str(DELTA_SLR[-1])):
            args['legend'] = True
            args['reverse_legend'] = True
            args['legend_outside'] = True
            args['legend_kwargs'] = dict(title='Turn')
        else:
            args['legend'] = False
    else:
        raise NotImplementedError('Invalid name: {}'.format(name))

    df.plot.area(cmap=ALTERNATING, legend=False)
    save_fig(path, **args)