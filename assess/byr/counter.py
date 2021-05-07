from assess.byr.util import bin_plot, save_dict
from featnames import X_OFFER, CON, INDEX, AUTO, EXP


def get_y(data=None, turn=None):
    assert turn in [3, 5]
    # buyer's offer
    byrcon = data[X_OFFER][CON].xs(turn, level=INDEX)
    # restrict to active seller counters
    df0 = data[X_OFFER].xs(turn - 1, level=INDEX)
    active = ~df0[AUTO] & ~df0[EXP]
    idx = byrcon.index.intersection(active[active].index)
    byrcon = byrcon.loc[idx]
    # Pr(counter)
    return (byrcon > 0) & (byrcon < 1)


def main():
    d3 = bin_plot(name='counter_3', get_y=lambda data: get_y(data=data, turn=3))
    d5 = bin_plot(name='counter_5', get_y=lambda data: get_y(data=data, turn=5))

    save_dict(d, 'counter')


if __name__ == '__main__':
    main()
