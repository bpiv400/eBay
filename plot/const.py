import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from viscid.plot import vpyplot as vlt

plt.style.use('seaborn-colorblind')


BIN_TICKS = [10, 20, 50, 100, 200, 500]
BIN_TICKS_SHORT = [10, 20, 30, 50, 100, 200, 300]
SLRBO_TICKS = [1, 10, 100, 1000, 10000]
FONTSIZE = 16
COLORS = ['k'] + vlt.get_current_colorcycle()
TRICOLOR = {'Humans': COLORS[0],
            'Agent': COLORS[1],
            'Impatient agent': COLORS[1],
            'Patient agent': COLORS[2],
            'Heuristic impatient agent': COLORS[1],
            'Heuristic patient agent': COLORS[2]}
ALTERNATING = ListedColormap([COLORS[int(i % 2) + 1] for i in range(1, 8)])
