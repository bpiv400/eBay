clear all
cd /data/eBay
use dta/clean

* add in BINs

rename (byr byr_us) =_thread
merge m:1 lstg using dta/listings, nogen update keepus(slr byr byr_us end_date bo *_price)
keep if thread != . | bo == 0

g double end_time = clock(string(end_date, "%td") + " 23:59:59", "DMYhms")
format %tc end_time
drop end_date

egen byte tag = tag(lstg)
g byte check = thread != . & bo == 0 & tag
drop tag bo

expand 2*check, gen(bin)
drop check
replace bin = 1 if thread == .

replace price = start_price if bin
replace message = . if bin
replace reject = 0 if bin
replace accept = 1 if bin

replace byr_thread = byr if bin
replace byr_us_thread = byr_us if bin
drop byr byr_us
rename *_thread *

replace thread = . if bin
replace index = . if bin
replace byr_hist = . if bin
replace clock = . if bin

* differentiate between BINs from new buyers and BINs in threads

by lstg byr, sort: egen byte count = sum(1)
replace bin = 0 if count > 1
drop count

* new bins

replace thread = 0 if bin
replace index = 1 if bin

* bins in threads: thread number and byr experience

gsort lstg byr -thread
by lstg byr: replace thread = thread[_n-1] if index == .
by lstg byr: replace byr_hist = byr_hist[_n-1] if index == .

* bins in threads: during buyer turn

sort lstg thread index
by lstg thread: replace index = index[_n-1] + 1 if index == . & mod(index[_n-1],2) == 0

* bins in threads: during seller turn

by lstg thread: replace clock = clock[_n-1] if index == .
by lstg thread: replace index = index[_n-1] if index == .

sort lstg thread index, stable
by lstg thread index: keep if _n == _N

* assume same-day non-expired reject on another thread is time of bin

g byte temp = thread if clock == .
by lstg: egen byte binthread = max(temp)
drop temp

g double temp = clock if thread != binthread
by lstg: egen double maxclock = max(temp)
drop temp binthread

by lstg thread: g byte exp = (clock - clock[_n-1]) / 1000 == 172800
g byte temp = reject & !exp if clock == maxclock
by lstg: egen byte check = max(temp)
drop temp exp

by lstg: egen double temp = max(clock)
replace check = 0 if temp > maxclock
drop temp

replace clock = maxclock if clock == . & dofc(maxclock) == dofc(end_time) & check
replace clock = end_time if clock == . & thread == 0
drop maxclock check

* bins in thread: randomly fill in missing clocks

by lstg: egen double last_clock = max(clock)
replace last_clock = max(last_clock, end_time + 1000 * (1 - 24 * 3600))
replace clock = last_clock + runiform() * (end_time - last_clock) if clock == .
drop last_clock

* save

save dta/bins, replace