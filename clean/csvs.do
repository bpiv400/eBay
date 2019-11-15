clear all
cd /data/eBay

*** offers

use dta/listings, clear

keep if unique
keep lstg

merge 1:m lstg using dta/offers, nogen keep(3)

format price %9.2f
export delim lstg-message using clean/offers.csv, dataf replace

*** threads

use dta/listings, clear

keep if unique
keep lstg

merge 1:m lstg using dta/threads, nogen keep(3)

* clean clock variables

drop end_time flag
format %9.0f start_time
replace start_time = ///
	(start_time - clock("01jun2012 00:00:00","DMYhms")) / 1000

export delim using clean/threads.csv, dataf replace

*** listings

use dta/listings, clear

keep if unique
keep lstg

merge 1:m lstg using dta/threads, nogen keep(3) keepus(flag end_time)

collapse (max) flag, by(lstg end_time)

merge 1:1 lstg using dta/listings, nogen
keep if unique
drop byr* views wtchrs bo sale_price relisted leaf product title unique

order flag, last
replace flag = 0 if flag == .
replace flag = 1 if bin_rev
drop bin_rev

* shipping

g byte fast = ship_fast != -1
drop ship_*

* reformat clock variables

order end_time, a(start_date)
replace end_time = clock(string(end_date + 1, "%td"), "DMY") - 1000 ////
	if end_time == .
drop end_date

format start_date end_time %9.0f
replace start_date = start_date - date("01jun2012","DMY")
replace end_time = ///
	(end_time - clock("01jun2012 00:00:00","DMYhms")) / 1000
replace end_time = round(end_time)
recast long end_time
	
* feedback score as int

g long temp = round(fdbk_score * fdbk_pstv)
order temp, a(fdbk_score)
drop fdbk_pstv
rename temp fdbk_pstv
	
* clean and save

format *price %9.2f
order lstg slr
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
