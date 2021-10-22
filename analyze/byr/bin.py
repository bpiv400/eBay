from analyze.byr.util import bin_plot
from analyze.util import save_dict
from featnames import X_OFFER, CON, INDEX


def get_y(data=None):
    con1 = data[X_OFFER][CON].xs(1, level=INDEX)
    assert not any(con1 == 0)
    acc1 = con1 == 1
    return acc1


def main():
    d = bin_plot(name='byrbin', get_y=get_y)
    save_dict(d, 'bin')


if __name__ == '__main__':
    main()
