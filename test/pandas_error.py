
import pandas as pd
from io import StringIO
t = "a, b, c, d\n1970-01-01 11:15, 2, 3, 4\n1970-01-01 11:15, 2, 3, 4\n, 2, 3, 4"
p = pd.read_csv(StringIO(t), parse_dates=['a'])
