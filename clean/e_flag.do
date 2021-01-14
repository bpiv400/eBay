clear all
cd /data/eBay
use dta/conform

* flag weird behavior with prices

g byte flag = 0

//replace flag = 1 if sale_price != start_price & bin
//replace flag = 1 if sale_price != price & accept
drop sale_price

replace flag = 1 if price > start_price
replace flag = 1 if price == start_price & !bin & index == 1

sort lstg thread index
by lstg thread: replace flag = 1 if price < price[_n-2] & _n > 2 & mod(_n,2) == 1
by lstg thread: replace flag = 1 if price > price[_n-2] & _n > 2 & mod(_n,2) == 0
by lstg thread: replace flag = 1 if price < price[_n-1] & mod(index,2) == 0
by lstg thread: replace flag = 1 if price > price[_n-1] & _n > 1 & mod(index,2) == 1

by lstg thread: replace flag = 1 if _n == _N & price < decline_price & mod(index,2) == 1 & !reject
by lstg thread: replace flag = 1 if _n < _N & price < decline_price & mod(index,2) == 1 & (clock != clock[_n+1] | !reject[_n+1])

by lstg thread: replace flag = 1 if _n == _N & price >= accept_price & mod(index,2) == 1 & !accept
by lstg thread: replace flag = 1 if _n < _N & price >= accept_price & mod(index,2) == 1 & (clock != clock[_n+1] | !accept[_n+1])

by lstg thread: replace flag = 1 if _n < _N & price < accept_price & price >= decline_price & clock == clock[_n+1]
drop start_price decline_price

replace flag = 1 if price == 0 & index == 1

* flag offers that come after 48 hours

sort lstg thread index
by lstg thread: replace flag = 1 if (clock - clock[_n-1]) / 1000 > 48 * 3600 & _n > 1

* flag offers at 48 hours that are not rejects

sort lstg thread index
by lstg thread: replace flag = 1 if (clock - clock[_n-1]) / 1000 == 48 * 3600 & !reject & _n > 1

* flag in-thread buyer bin offers before seller responds

sort lstg thread index
by lstg thread: replace flag = 1 if clock == clock[_n-1] & _n > 1 & mod(index, 2) == 1

* flag 7th-turn offers that are neither accepts nor rejects

replace flag = 1 if index == 7 & !accept & !reject

* flag listings that expire before a week

sort lstg thread index
by lstg: egen byte sells = max(accept)

merge m:1 lstg using dta/listings, nogen keep(3) keepus(start_date)
g int days = dofc(end_time) - start_date
replace flag = 1 if days < 7 & !sells
drop start_date days sells

* flag entire listing

sort lstg thread index
by lstg: egen byte temp = max(flag)
replace flag = temp
drop temp

* fill in missing variables

sort lstg thread index
by lstg thread: egen double start_time = min(clock)
foreach x in byr slr {
	sort `x' start_time lstg thread index
	by `x': replace `x'_hist = `x'_hist[_n-1] if `x'_hist == . & _n > 1
	replace `x'_hist = 0 if `x'_hist == .
}
drop slr start_time

replace message = 0 if message == .

* save offers

sort lstg thread index
order lstg thread index clock price accept reject message bin flag byr* end_time
savesome lstg-message if !flag using dta/offers, replace

* save threads

collapse (max) bin (max) byr_us, by(lstg thread byr byr_hist end_time flag)
	
replace byr_hist = byr_hist - (1-bin)

save dta/threads, replace
