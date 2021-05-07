from agent.util import get_sim_dir, only_byr_agent, load_valid_data
from agent.const import DELTA_BYR
from featnames import CON, X_OFFER


def main():
    for delta in DELTA_BYR:
        sim_dir = get_sim_dir(byr=True, delta=delta)
        data = only_byr_agent(load_valid_data(sim_dir=sim_dir, minimal=True))
        con = data[X_OFFER][CON].unstack()[[1, 3, 5]]
        for t in con.columns:
            con = con[con[t] == con[t].mode().item()]
        print(con.mean())


if __name__ == '__main__':
    main()
