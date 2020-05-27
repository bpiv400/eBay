clear all
cd /data/eBay
import delim using raw/listings.csv

* rename variables

foreach x in _id _start _usd _code _categ est {
	rename *`x' *
}

foreach x in anon_ item_ time_ {
	rename *`x'* **
}
rename *buyer* *byr*

rename *count* *ct*
rename *_ct *s
rename (to_lst_cnt bo_lst_cnt) (slr_lstg_ct slr_bo_ct)
rename lstg_gen_type relisted
rename price sale_price
rename item lstg
rename bo_ck_yn bo
	
* convert start and end to ints
	
g int start_date = date(auct_start_dt,"DMY")
g int end_date = date(auct_end_dt,"DMY")
format *_date %td
drop auct_*_dt

* encode variables

do ~/Dropbox/eBay/repo/clean/encode.do

* leaf starts at 0

replace leaf = leaf - 1

* fill in accept/decline prices

replace accept_price = start_price if accept_price == . | accept_price == 0
replace decline_price = 0 if decline_price == .

* add to bin_rev field

replace bin_rev = 1 if bo == 0 & sale_price != start_price
replace bin_rev = 1 if decline_price > start_price
replace bin_rev = 1 if accept_price > start_price
replace bin_rev = 1 if decline_price > accept_price
replace decline_price = start_price if decline_price > start_price
replace accept_price = start_price if accept_price > start_price

* feedback variables

replace fdbk_score = 0 if fdbk_score == .
replace fdbk_pstv = fdbk_pstv / 100
replace fdbk_pstv = 1 if fdbk_pstv == .

* title occurs only once?

by title, sort: egen int temp = sum(1)
g byte unique = temp == 1
drop temp title

* listings to keep

keep if unique
drop unique

keep if !bin_rev
drop bin_rev

g long cents = round(start_price * 100)
keep if cents <= 1000 * 100 & cents >= 9.95 * 100
drop cents

drop if meta == 316 | meta == 6000 | meta == 10542 | meta == 172008

* create single category

replace product = . if product == 547957
by product, sort: egen long count = sum(1)

g cat = "p" + string(product) if count >= 1000 & product != .
replace cat = "l" + string(leaf) if cat == ""
drop count

by cat, sort: egen long count = sum(1)
replace cat = "m" + string(meta) if count < 1000
drop count

* feedback score as int

g long temp = round(fdbk_score * fdbk_pstv)
order temp, a(fdbk_score)
drop fdbk_pstv
rename temp fdbk_pstv

* save

drop ct? ref_price?
order lstg slr cat meta leaf product cndtn start_date end relisted
sort lstg
save dta/listings, replace
