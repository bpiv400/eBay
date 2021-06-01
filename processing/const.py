from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar


# date range and holidays
START = '2012-06-01 00:00:00'
END = '2013-05-31 23:59:59'
HOLIDAYS = Calendar().holidays(start=START, end=END)

# fixed random seed
SEED = 123456
