from assess.byr.util import bin_plot
from featnames import X_OFFER, CON


def get_y(data=None):
    con = data[X_OFFER][CON].unstack()[[1, 3, 5]]
    count = ((0 < con) & (con < 1)).sum(axis=1)
    count = count[count > 0]
    return count


def main():
    bin_plot(name='offers', get_y=get_y)


if __name__ == '__main__':
    main()
