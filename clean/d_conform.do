clear all
cd /data/eBay
use dta/bins

* recode accepts and rejects

sort lstg thread index
by lstg thread: replace accept = price == price[_n-1] | bin
by lstg thread: replace reject = price == price[_n-2] | (price == start_price & index == 2)

* find time of first accept

g double temp = clock if accept
by lstg: egen double accept_time = min(temp)
replace end_time = min(end_time, accept_time)
drop temp accept_time

* delete offers in thread after accept

g byte temp = index if accept
by lstg thread: egen byte idx = min(temp)
drop if index > idx
drop temp idx

* delete offers after time of first accept

drop if clock > end_time

* delete offers in other threads at or after time of first accept

g byte temp = thread if accept
by lstg: egen byte accept_thread = min(temp)
drop if clock == end_time & thread != accept_thread
drop temp accept_thread

* delete reject when buyer does not return

sort lstg thread index
by lstg thread: drop if _n == _N & mod(index,2) == 1 & index < 7 & reject & (clock - clock[_n-1]) / 1000 == 172800

* delete thread activity after byr reject

sort lstg thread index
by lstg thread: g byte check = (index == 3 | index == 5) & reject
g byte temp = index if check
by lstg thread: egen byte temp2 = min(temp)
drop if index > temp2
drop check temp*

* delete activity after 31 days

merge m:1 lstg using dta/listings, nogen keep(3) keepus(*_date)
format clock %tc

replace end_date = start_date + 30 if end_date > start_date + 30
g double new_end_time = clock(string(end_date, "%td") + " 23:59:59", "DMYhms")
replace end_time = new_end_time if end_time > new_end_time
drop new_end_time *_date
drop if clock > end_time

* add expired rejects when buyer does not return

sort lstg thread index
by lstg thread: g byte check = !accept & _n == _N & mod(index,2) == 0
expand 2*check, gen(copy)
drop check

replace index = index + 1 if copy
replace reject = 1 if copy
replace message = . if copy

sort lstg thread index
by lstg thread: replace clock = min(end_time, clock[_n-1] + 48 * 3600 * 1000) if copy
by lstg thread: replace price = price[_n-2] if copy

g byte censored = copy & clock == end_time
drop copy

* add expired rejects when listing ends or sells on different thread

by lstg thread: g byte check = !accept & _n == _N & mod(index,2) == 1 & price != price[_n-2]
expand 2*check, gen(copy)
drop check

replace index = index + 1 if copy
replace reject = 1 if copy
replace message = . if copy

sort lstg thread index
by lstg thread: replace clock = end_time if copy
by lstg thread: replace price = price[_n-2] if copy & index > 2
by lstg thread: replace price = start_price if copy & index == 2

replace censored = 1 if copy
drop copy

* renumber threads

g byte new = index == 1
g double temp = clock if new
by lstg thread, sort: egen double thread_start = min(temp)
drop temp

sort lstg thread_start thread index
by lstg: g int newid = sum(new)
order newid, a(thread)
drop new thread_start thread
rename newid thread

* save

save dta/conform, replace