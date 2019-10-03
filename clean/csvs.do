clear all
cd ~/Dropbox/eBay/data

* offers

use dta/offers, clear

format price %9.2f
export delim lstg-message using clean/offers.csv, dataf replace

* threads

use dta/threads, clear

drop end_time flag
format %9.0f start_time
replace start_time = ///
	(start_time - clock("01jun2012 00:00:00","DMYhms")) / 1000
export delim using clean/threads.csv, dataf replace

* listings

use dta/threads, clear

collapse (max) flag, by(lstg end_time)

merge 1:1 lstg using dta/listings, nogen
drop byr* sale_price views wtchrs bo relisted

replace product = 0 if product == 547957

g byte fast = ship_fast != -1
drop ship_*

order flag, last
replace flag = 0 if flag == .
replace flag = 1 if bin_rev
drop bin_rev

order end_time, a(start_date)
replace end_time = clock(string(end_date + 1, "%td"), "DMY") - 1000 ////
	if end_time == .
drop end_date

format start_date end_time %9.0f
replace start_date = start_date - date("01jun2012","DMY")
replace end_time = ///
	(end_time - clock("01jun2012 00:00:00","DMYhms")) / 1000

format fdbk_pstv *price %9.2f
order slr lstg
sort slr lstg
export delim using clean/listings.csv, nolab dataf replace
