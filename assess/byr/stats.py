from agent.util import get_sale_norm, load_valid_data
from featnames import X_OFFER, LOOKUP, START_PRICE, CON, NORM, INDEX


def main():
    data = load_valid_data(byr=True, minimal=True, clock=True)

    # negotiated sale price
    sale_norm = get_sale_norm(offers=data[X_OFFER])
    sale_norm = sale_norm[sale_norm < 1]
    start_price = data[LOOKUP][START_PRICE].loc[sale_norm.index]

    print('Average negotiated sale price: {}'.format(sale_norm.mean()))
    print('Average negotiated sale price for items priced around $10: {}'.format(
        sale_norm[start_price <= 10].mean()))
    print('Average negotiated sale price for items priced around $500: {}'.format(
        sale_norm[(start_price >= 495) & (start_price <= 500)].mean()))

    # response to first offer
    norm = data[X_OFFER][NORM].unstack()[[1, 2]]
    norm = norm.loc[norm[1] < 1, :]
    print('Share of first offers that don\'t get a response: {}'.format(
        norm[2].isna().mean()))
    print('Average discount in response to first offer: {}'.format(norm[2].mean()))

    # walk rate on turn 3
    con3 = data[X_OFFER][CON].xs(3, level=INDEX)
    print('Walk rate on turn 3: {}'.format((con3 == 0).mean()))


if __name__ == '__main__':
    main()
