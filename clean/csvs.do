clear all
cd /data/eBay

* offers

use dta/offers, clear

format price %9.2f
export delim lstg-message using clean/offers.csv, dataf replace

*** threads

use dta/threads, clear

* clean clock variables

drop end_time flag
format %9.0f start_time
replace start_time = ///
	(start_time - clock("01jun2012 00:00:00","DMYhms")) / 1000
	
export delim using clean/threads.csv, dataf replace

*** listings

* merge in flag

use dta/threads, clear

collapse (max) flag, by(lstg end_time)

merge 1:1 lstg using dta/listings, nogen
drop byr* views wtchrs bo sale_price relisted

order flag, last
replace flag = 0 if flag == .
replace flag = 1 if bin_rev
drop bin_rev

* shipping

g byte fast = ship_fast != -1
drop ship_*

* create single category

replace product = . if product == 547957
by product, sort: egen long count = sum(1)

g cat = "p" + string(product) if count >= 1000 & product != .
replace cat = "l" + string(leaf) if cat == ""
drop count leaf product

by cat, sort: egen count = sum(1)
replace cat = "m" + string(meta) if count < 1000
drop count
order cat, a(meta)

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
