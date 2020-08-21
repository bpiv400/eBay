clear all
cd ~/weka/eBay

*** threads

use dta/threads, clear
drop if flag
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

use dta/threads, clear
keep lstg flag end_time
sort lstg flag end_time
by lstg flag end_time: keep if _n == 1

merge 1:1 lstg using dta/listings, nogen
keep if flag != 1
drop byr* views wtchrs bo sale_price flag

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

*** seller sentences for word2vec (use flagged listings)

use dta/listings, clear

keep slr leaf
sort slr leaf
by slr leaf: keep if _n == 1

by slr: egen long count = sum(1)
drop if count == 1
drop count

export delim using clean/leaf_slr.csv, dataf replace

*** buyer sentences for word2vec (use flagged listings)

use dta/threads, clear
keep lstg byr

merge m:1 lstg using dta/listings, nogen keep(3) keepus(leaf)
drop lstg

sort byr leaf
by byr leaf: keep if _n == 1
by byr: egen long count = sum(1)
drop if count == 1
drop count

export delim using clean/leaf_byr.csv, dataf replace

*** meta names

use dta/listings, clear
keep meta
sort meta
by meta: keep if _n == 1
decode meta, gen(label)
label drop meta

export delim using clean/meta.csv, dataf replace