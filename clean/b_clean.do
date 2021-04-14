clear all
cd /data/eBay/
import delim using raw/threads.csv

rename anon_* *
rename *_id *
rename offr_* *
rename any_mssg message
rename item lstg
drop src_cre_dt fdbk_*

* for word2vec features

merge m:1 lstg using dta/w2v_slr, nogen keep(3) keepus(leaf)
savesome byr leaf using dta/w2v_byr, replace
drop leaf

* clean clock

g double clock = clock(src_cre_date, "DMYhms")
g long seconds = (clock(response_time,"DMYhms") - clock) / 1000 if status != 7
drop src_cre_date response_time
format clock %tc

sort slr lstg byr thread clock
by slr lstg byr thread: replace seconds = (clock[_n+1] - clock) / 1000 if status == 7

* double entries

collapse (min) *_hist, by(slr lstg byr thread clock seconds type status price message byr_us)

* double entries with different responses (keep first)

sort slr lstg byr thread clock type price seconds
by slr lstg byr thread clock type price: keep if _n == 1

* coterminous offers by same buyer on same thread (keep max)

sort slr lstg byr thread clock type price
by slr lstg byr thread clock type: keep if _n == _N

* double threads with identical beginning (keep longer)

by slr lstg byr thread, sort: egen byte length = sum(1)
by slr lstg byr, sort: egen byte maxlength = max(length)
keep if length == maxlength
drop *length

* double threads with identical attributes

collapse (min) thread (min) seconds, by(slr lstg byr clock type status price message byr_us *_hist)

* double threads with identical beginning (keep first response)

sort slr lstg byr clock seconds
by slr lstg byr clock: drop if _n > 1

* double threads one second apart (keep first)

sort slr lstg byr price clock
by slr lstg byr price: drop if clock - clock[_n-1] == 1000 & thread != thread[_n-1]

* double entries one second apart (keep first)

sort slr lstg byr thread type status price clock
by slr lstg byr thread type status price: drop if clock - clock[_n-1] == 1000

* make byr_us thread-specific

by slr lstg byr thread, sort: egen temp = max(byr_us)
replace byr_us = temp
drop temp

* common hist count within thread

foreach var of varlist ???_hist {
	by slr lstg byr thread, sort: egen temp = min(`var')
	replace `var' = temp
	drop temp
}

* offer index

sort slr lstg byr thread clock
by slr lstg byr thread: g byte index = 1 if _n == 1
by slr lstg byr thread: replace index = index[_n-1] + 1 * (type == 2 & type[_n-1] == 0) + 1 * (type == 1 & type[_n-1] == 2) + 1 * (type == 2 & type[_n-1] == 1) + 1 * (type == 0 & type[_n-1] == 2) + 2 * (type == 0 & type[_n-1] == 1) + 2 * (type == 0 & type[_n-1] == 0) if _n > 1
drop slr

* new thread id

sort lstg thread index
by lstg: g int newid = 1 if _n == 1
by lstg: replace newid = newid[_n-1] * (thread == thread[_n-1]) + (newid[_n-1] + 1) * (thread != thread[_n-1]) if _n > 1
drop thread
rename newid thread

* drop offers after two week lapse

sort lstg thread index
by lstg thread: g double temp = 0 if _n == 1
by lstg thread: replace temp = max(temp[_n-1], (clock - clock[_n-1])/1000) if _n > 1
drop if temp >= 14 * 24 * 3600
drop temp

* reshape

reshape wide clock type status price message seconds, i(lstg thread) j(index)
g double clock7 = .
reshape long

* drop non-existent indices

sort lstg thread index
by lstg thread: replace clock = clock[_n-1] + 1000 * seconds[_n-1] if clock == .
drop if clock == .
drop seconds

* save clock

format %tc clock
save clean/clock, replace

* restrict lstgs and merge in start_price

merge m:1 lstg using dta/listings, nogen keep(3) keepus(start_price)
drop byr_hist

* clean prices

sort lstg thread index
by lstg thread: g byte reject = type == . & (status[_n-1] == 0 | status[_n-1] == 2 | status[_n-1] == 6 | status[_n-1] == 8)
by lstg thread: replace reject = 1 if type == 2 & (price == start_price | price == price[_n-2])
by lstg thread: g byte accept = status[_n-1] == 1 | status[_n-1] == 9
replace price = start_price if price == . & index == 2 & reject
by lstg thread: replace price = price[_n-1] if price == . & accept
by lstg thread: replace price = price[_n-2] if price == . & reject

* clean seller auto-response

sort lstg thread index
by lstg thread: g long sec = (clock - clock[_n-1]) / 1000
by lstg thread: g byte auto = status[_n-1] == 6 | status[_n-1] == 9
by lstg thread: replace auto = 1 if mod(index,2) == 0 & reject & start_price == price & (sec <= 0 | sec[_n+1] < 0)
by lstg thread: replace clock = clock[_n-1] if auto
drop auto type status

* clean (pre-)instantaneous buyer offers

by lstg thread: replace sec = (clock - clock[_n-1]) / 1000
by lstg thread: replace clock = clock[_n-1] + 1000 * 172800 if _n == _N & mod(index,2) == 1 & sec <= 0 & reject

* negative seconds

by lstg thread: replace sec = (clock - clock[_n-1]) / 1000
by lstg thread: replace clock = clock[_n-1] + 60 * 1000 if _n > 3 & (sec < 0 | sec[_n+1] < 0)

* expired offers

by lstg thread: replace sec = (clock - clock[_n-1]) / 1000
by lstg thread: replace clock = clock + 1 * 1000 if reject & sec == 172799
by lstg thread: replace clock = clock + 2 * 1000 if reject & sec == 172798
by lstg thread: replace clock = clock[_n-1] + 172800 * 1000 if reject & sec > 172800 & sec != .

by lstg thread: replace sec = (clock - clock[_n-1]) / 1000
by lstg thread: replace clock = clock[_n-1] + 172799 * 1000 if mod(index,2) == 0 & accept & sec > 172800 & sec != .
drop sec

* save

save dta/clean, replace
