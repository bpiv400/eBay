clear all
cd /data/eBay/

* listings

import delim using raw/listings.csv
rename anon_*_id *
rename auct_*_dt *_date
rename (buyer item) (byr lstg)
keep lstg slr byr *_date

foreach var in start_date end_date {
	g int temp = date(`var',"DMY")
	drop `var'
	rename temp `var'
}
format *_date %td
drop start_date

tempfile temp
save `temp'

* threads

use clean/clock, clear

keep if index == 1
keep lstg byr byr_hist clock
merge m:1 lstg using `temp', nogen keep(1 3) keepus(slr)
order lstg slr

* join

append using `temp'
drop if byr == .

sort lstg slr byr byr_hist
by lstg slr byr: keep if _n == 1

replace clock = clock(string(end_date + 1, "%td"), "DMY") - 1000 if clock == .
g byte bin = end_date != .
drop end_date

replace byr_hist = byr_hist - 1

sort byr clock byr_hist
by byr: replace byr_hist = byr_hist[_n-1] if byr_hist == .

gsort byr -clock
by byr: replace byr_hist = byr_hist[_n-1] if byr_hist == .

g byte missing = byr_hist == .
replace byr_hist = 0 if missing

sort byr clock
save clean/hist, replace

* unique interactions

use clean/hist, clear

collapse (count) count=lstg, by(slr byr)

g byte unique = count == 1
tab unique

collapse (sum) count, by(unique)
