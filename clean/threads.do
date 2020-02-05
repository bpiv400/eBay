clear all
cd /data/eBay
import delim using raw/threads.csv

rename anon_* *
rename *_id *
rename offr_* *
rename any_mssg message
rename item lstg
drop src_cre_dt fdbk_*

* restrict to unique lstgs

merge m:1 lstg using dta/listings, nogen keep(3) keepus(unique)
keep if unique
drop unique
	
* clean clock

g double clock = clock(src_cre_date,"DMYhms")
g long seconds = (clock(response_time,"DMYhms") - clock) / 1000 ///
	if status != 7
drop src_cre_date response_time	
format clock %tc

sort slr lstg byr thread clock
by slr lstg byr thread: replace seconds = ///
	(clock[_n+1] - clock) / 1000 if status == 7
	
* replace missing messages

replace message = 0 if message == .

* save temp

save dta/temp0, replace	

* double entries

collapse (min) *_hist, by(slr lstg byr thread clock seconds ///
	type status price message byr_us)
	
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

collapse (min) thread (min) seconds, by(slr lstg byr clock ///
	type status price message byr_us *_hist)
	
* double threads with identical beginning (keep first response)

sort slr lstg byr clock seconds
by slr lstg byr clock: drop if _n > 1

* double threads one second apart (keep first)

sort slr lstg byr price clock
by slr lstg byr price: drop if clock - clock[_n-1] == 1000 ///
	& thread != thread[_n-1]
	
* double entries one second apart (keep first)

sort slr lstg byr thread type status price clock
by slr lstg byr thread type status price: ///
	drop if clock - clock[_n-1] == 1000
	
* make byr_us thread-specific

by slr lstg byr thread, sort: egen temp = max(byr_us)
replace byr_us = temp
drop temp

foreach var of varlist ???_hist {
	by slr lstg byr thread, sort: egen temp = min(`var')
	replace `var' = temp
	drop temp
}
	
* offer index

sort slr lstg byr thread clock
by slr lstg byr thread: g byte index = 1 if _n == 1
by slr lstg byr thread: replace index = index[_n-1] + ///
	1 * (type == 2 & type[_n-1] == 0) + ///
	1 * (type == 1 & type[_n-1] == 2) + ///
	1 * (type == 2 & type[_n-1] == 1) + ///
	1 * (type == 0 & type[_n-1] == 2) + ///
	2 * (type == 0 & type[_n-1] == 1) + ///
	2 * (type == 0 & type[_n-1] == 0) if _n > 1
drop slr

* new thread id

sort lstg thread index
by lstg: g int newid = 1 if _n == 1
by lstg: replace newid = newid[_n-1] * (thread == thread[_n-1]) ///
	+ (newid[_n-1] + 1) * (thread != thread[_n-1]) if _n > 1
drop thread
rename newid thread

* save temp

save dta/temp1, replace

* drop offers after two week lapse

sort lstg thread index
by lstg thread: g double temp = 0 if _n == 1
by lstg thread: replace temp = max(temp[_n-1], ///
	(clock - clock[_n-1])/1000) if _n > 1
drop if temp >= 14 * 24 * 3600
drop temp

* reshape

reshape wide clock type status price message seconds, ///
	i(lstg thread) j(index)
g double clock7 = .
reshape long

* drop non-existent indices

sort lstg thread index
by lstg thread: replace clock = clock[_n-1] + 1000 * seconds[_n-1] ///
	if clock == .
drop if clock == .	
drop seconds

* merge in start_price from listings

merge m:1 lstg using dta/listings, nogen keep(3) keepus(start_price)

* clean prices

sort lstg thread index
by lstg thread: g byte reject = type == . & (status[_n-1] == 0 | ///
	status[_n-1] == 2 | status[_n-1] == 6 | status[_n-1] == 8)
by lstg thread: replace reject = 1 if type == 2 & ///
	(price == start_price | price == price[_n-2])
by lstg thread: g byte accept = status[_n-1] == 1 | status[_n-1] == 9
replace price = start_price if price == . & index == 2 & reject
by lstg thread: replace price = price[_n-1] if price == . & accept
by lstg thread: replace price = price[_n-2] if price == . & reject
drop start_price

* save temp

save dta/temp2, replace

* clean seller auto-response

sort lstg thread index
by lstg thread: g long sec = (clock - clock[_n-1]) / 1000
by lstg thread: g byte auto = status[_n-1] == 6 | status[_n-1] == 9
by lstg thread: replace auto = 1 if mod(index,2) == 0 & reject & ///
	start_price == price & (sec <= 0 | sec[_n+1] < 0)
by lstg thread: replace clock = clock[_n-1] if auto
drop auto type status

* clean (pre-)instantaneous buyer offers

by lstg thread: replace sec = (clock - clock[_n-1]) / 1000
by lstg thread: replace clock = clock[_n-1] + 1000 * 172800 ///
	if _n == _N & mod(index,2) == 1 & sec <= 0 & reject

* negative seconds

by lstg thread: replace sec = (clock - clock[_n-1]) / 1000
by lstg thread: replace clock = clock[_n-1] + 60 * 1000 ///
	if _n > 3 & (sec < 0 | sec[_n+1] < 0)

* expired offers

by lstg thread: replace sec = (clock - clock[_n-1]) / 1000
by lstg thread: replace clock = clock + 1 * 1000 ///
	if reject & sec == 172799
by lstg thread: replace clock = clock + 2 * 1000 ///
	if reject & sec == 172798
by lstg thread: replace clock = clock[_n-1] + 172800 * 1000 ///
	if reject & sec > 172800 & sec != .
	
by lstg thread: replace sec = (clock - clock[_n-1]) / 1000
by lstg thread: replace clock = clock[_n-1] + 172799 * 1000 ///
	if mod(index,2) == 0 & accept & sec > 172800 & sec != .
drop sec

* save temp

save dta/temp2a, replace

* add in BINs

rename (byr byr_us) =_thread
merge m:1 lstg using dta/listings, nogen update ///
	keepus(slr byr byr_us end_date bo *_price)
keep if thread != . | bo == 0

g double end_time = ///
	clock(string(end_date, "%td") + " 23:59:59", "DMYhms")
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
replace clock = end_time if bin

* bins in threads

gsort lstg byr -thread
by lstg byr: replace thread = thread[_n-1] if index == .
by lstg byr: replace byr_hist = byr_hist[_n-1] if index == .

by lstg: egen double last_clock = max(clock)
replace last_clock = max(last_clock, end_time + 1000 * (1 - 24 * 3600))
replace clock = last_clock + runiform() * (end_time - last_clock) ///
	if index == .
drop last_clock

sort lstg thread index
by lstg thread: replace index = index[_n-1] + 1 ///
	if index == . & mod(index[_n-1],2) == 0
by lstg thread: replace index = index[_n-1] ///
	if index == . & mod(index[_n-1],2) == 1

sort lstg thread index, stable
by lstg thread index: keep if _n == _N

* save temp

save dta/temp2b, replace

* recode accepts and rejects

sort lstg thread index
by lstg thread: replace accept = price == price[_n-1] | bin
by lstg thread: replace reject = price == price[_n-2] | ///
	(price == start_price & index == 2)

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
by lstg thread: drop if _n == _N & mod(index,2) == 1 & ///
	index < 7 & reject & (clock - clock[_n-1]) / 1000 == 172800
	
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
g double new_end_time = ///
	clock(string(end_date, "%td") + " 23:59:59", "DMYhms")
replace end_time = new_end_time if end_time > new_end_time
drop new_end_time *_date
drop if clock > end_time
	
* save temp

save dta/temp2c, replace

* add expired rejects when buyer does not return

sort lstg thread index
by lstg thread: g byte check = !accept & _n == _N & mod(index,2) == 0
expand 2*check, gen(copy)
drop check

replace index = index + 1 if copy
replace reject = 1 if copy
replace message = . if copy

sort lstg thread index
by lstg thread: replace clock = min(end_time, ///
	clock[_n-1] + 14 * 24 * 3600 * 1000) if copy & index < 7
by lstg thread: replace clock = min(end_time, ///
	clock[_n-1] + 2 * 24 * 3600 * 1000) if copy & index == 7
by lstg thread: replace price = price[_n-2] if copy

g byte censored = copy & clock == end_time
drop copy

* add expired rejects when listing ends or sells on different thread

by lstg thread: g byte check = !accept & _n == _N & ///
	mod(index,2) == 1 & price != price[_n-2]
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

* save temp

save dta/temp2d, replace

* flag weird behavior with prices

g byte flag = 0

replace flag = 1 if sale_price != start_price & bin
replace flag = 1 if sale_price != price & accept
drop sale_price

replace flag = 1 if price > start_price
replace flag = 1 if price == start_price & !bin & index == 1
drop start_price

sort lstg thread index
by lstg thread: replace flag = 1 ///
	if price < price[_n-2] & _n > 2 & mod(_n,2) == 1
by lstg thread: replace flag = 1 ///
	if price > price[_n-2] & _n > 2 & mod(_n,2) == 0
by lstg thread: replace flag = 1 ///
	if price < price[_n-1] & mod(index,2) == 0
by lstg thread: replace flag = 1 ///
	if price > price[_n-1] & _n > 1 & mod(index,2) == 1
	
by lstg thread: replace flag = 1 if _n < _N & ///
	price >= accept_price & !bin & mod(index,2) == 1 & ///
	(clock != clock[_n+1] | !accept[_n+1])
by lstg thread: replace flag = 1 if _n < _N & ///
	price <= decline_price & mod(index,2) == 1 & ///
	(clock != clock[_n+1] | !reject[_n+1])
by lstg thread: replace flag = 1 if _n < _N & ///
	price < accept_price & price > decline_price & ///
	clock == clock[_n+1]
drop accept_price decline_price

* flag byr bin offers after more than a two-week delay

sort lstg thread index
by lstg thread: g long diff = (clock - clock[_n-1]) / 1000
replace flag = 1 if diff != . & diff > 14 * 24 * 3600
replace flag = 1 if index == 7 & diff != . & diff > 2 * 24 * 3600
drop diff

* fill in missing variables

by lstg thread: egen double start_time = min(clock)
foreach x in byr slr {
	sort `x' start_time lstg thread index
	by `x': replace `x'_hist = `x'_hist[_n-1] if `x'_hist == . & _n > 1
	replace `x'_hist = 0 if `x'_hist == .
}
drop slr start_time

replace message = 0 if message == .

* throw out bins from non-unique listings

merge m:1 lstg using dta/listings, nogen keep(3) keepus(unique)
keep if unique
drop unique

* save temp

save dta/temp3, replace

* save offers

sort lstg thread index
order lstg thread index clock price accept reject censored message ///
	bin flag byr* end_time
savesome lstg-message using dta/offers, replace

* save threads

collapse (max) bin (max) flag (max) byr_us, ///
	by(lstg thread byr byr_hist end_time)
	
replace byr_hist = byr_hist - (1-bin)

save dta/threads, replace
