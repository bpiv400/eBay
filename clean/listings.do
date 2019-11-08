clear all
cd /data/eBay
import delim using raw/listings.csv

* rename variables

foreach x in _id _start _usd _code _categ est {
	rename *`x' *
}

foreach x in anon_ buyer item_ time_ {
	rename *`x'* **
}

rename *count* *ct*
rename *_ct *s
rename (to_lst_cnt bo_lst_cnt lstg price item bo) ///
	(slr_lstg_ct slr_bo_ct relisted sale_price lstg bo)
	
* convert start and end to ints
	
g int start_date = date(auct_start_dt,"DMY")
g int end_date = date(auct_end_dt,"DMY")
format *_date %td
drop auct_*_dt

* encode variables

replace cndtn = "Not listed" if cndtn == ""
label def cndtn ///
	0 "Not listed" ///
	1 "New" ///
	2 "New other" ///
	3 "New with defects" ///
	4 "Manuf. refurbished" ///
	5 "Seller refurbished" ///
	6 "Like New" ///
	7 "Used" ///
	8 "Very Good" ///
	9 "Good" ///
	10 "Acceptable" ///
	11 "For parts", replace

label def meta 0 "Everything Else" ///
	1 "Collectibles" ///
	2 "Toys & Hobbies" ///
	3 "Dolls & Bears" ///
	4 "Stamps" ///
	5 "Books" ///
	6 "Jewelry & Watches" ///
	7 "Consumer Electronics" ///
	8 "Specialty Services" ///
	9 "Art" ///
	10 "Musical Instruments & Gear" ///
	11 "Cameras & Photo" ///
	12 "Pottery & Glass" ///
	13 "Sporting Goods" ///
	14 "Video Games & Consoles" ///
	15 "Pet Supplies" ///
	16 "Tickets & Experiences" ///
	17 "Baby" ///
	18 "Travel" ///
	19 "eBay Motors" ///
	20 "Real Estate" ///
	21 "Coins & Paper Money" ///
	22 "DVDs & Movies" ///
	23 "Music" ///
	24 "Clothing Shoes & Accessories" ///
	25 "Home & Garden" ///
	26 "Business & Industrial" ///
	27 "Crafts" ///
	28 "Cell Phones & Accessories" ///
	29 "Antiques" ///
	30 "Health & Beauty" ///
	31 "Entertainment Memorabilia" ///
	32 "Computers/Tablets & Networking" ///
	33 "Sports Mem Cards & Fan Shop" ///
	34 "Gift Cards & Coupons", replace

foreach var of varlist meta cndtn {
	encode `var', gen(`var') label(`var') noextend
}

replace leaf = leaf - 1

* fill in accept/decline prices

replace accept_price = start_price if accept_price < decline_price
replace accept_price = start_price if accept_price == .
replace decline_price = 0 if decline_price == .
replace decline_price = start_price if decline_price > start_price
replace accept_price = start_price if accept_price > start_price

* add to bin_rev field

replace bin_rev = 1 if bo == 0 & sale_price != start_price

* feedback variables

replace fdbk_score = 0 if fdbk_score == .
replace fdbk_pstv = fdbk_pstv / 100
replace fdbk_pstv = 1 if fdbk_pstv == .

* save

drop ct? ref_price?
order lstg slr meta leaf product title cndtn start end relisted
sort lstg
save dta/listings, replace
