clear all
cd /data/eBay

*** threads

use dta/threads, clear
drop end_time flag

export delim using clean/threads.csv, dataf replace

*** offers

use dta/offers, clear

* clean clock variables

format %9.0f clock
replace clock = (clock - clock("01jun2012 00:00:00","DMYhms")) / 1000

format price %9.2f
export delim using clean/offers.csv, dataf replace

*** listings

use dta/offers, clear

collapse (sum) accept, by(lstg thread)

merge 1:1 lstg thread using dta/threads, nogen keepus(flag end_time)

collapse (max) flag (sum) accept, by(lstg end_time)

merge 1:1 lstg using dta/listings, nogen
keep if unique
drop byr* views wtchrs bo sale_price leaf product title unique

* flag listings in which start_price has changed

order flag, last
replace flag = 0 if flag == .
replace flag = 1 if bin_rev
drop bin_rev

* flag listings that expired without sale in fewer than 31 days

replace flag = 1 if !accept & end_date - start_date < 30
drop accept

* shipping

g byte fast = ship_fast != -1
drop ship_*

* reformat clock variables

order end_time, a(start_date)
replace end_date = start_date + 30 if end_date > start_date + 30
replace end_time = clock(string(end_date + 1, "%td"), "DMY") - 1000 if end_time == .
drop end_date

format start_date end_time %9.0f
replace start_date = start_date - date("01jun2012","DMY")
replace end_time = (end_time - clock("01jun2012 00:00:00","DMYhms")) / 1000
replace end_time = round(end_time)
recast long end_time
	
* clean and save

format *price %9.2f
sort lstg slr
export delim using clean/listings.csv, nolab dataf replace

*** seller sentences for word2vec

use dta/listings, clear
keep slr cat
order slr cat

sort slr cat
by slr cat: keep if _n == 1
by slr: egen long count = sum(1)
drop if count == 1
drop count

export delim using clean/cat_slr.csv, dataf replace

*** buyer sentences for word2vec

use dta/threads, clear
keep lstg byr

merge m:1 lstg using dta/listings, nogen keep(3) keepus(cat)
drop lstg

sort byr cat
by byr cat: keep if _n == 1
by byr: egen long count = sum(1)
drop if count == 1
drop count

export delim using clean/cat_byr.csv, dataf replace
