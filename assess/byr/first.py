from assess.byr.util import bin_plot, save_dict
from featnames import X_OFFER, CON, INDEX


def get_y(data=None):
    con1 = data[X_OFFER][CON].xs(1, level=INDEX)
    con1 = con1[(0 < con1) & (con1 < 1)]
    return con1


def main():
    d = bin_plot(name='first', get_y=get_y)
    save_dict(d, 'first')


if __name__ == '__main__':
    main()
