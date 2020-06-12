replace cndtn = 0 if cndtn == .
replace cndtn = 1 if cndtn == 1000
replace cndtn = 2 if cndtn == 1500
replace cndtn = 3 if cndtn == 1750
replace cndtn = 4 if cndtn == 2000
replace cndtn = 5 if cndtn == 2500
replace cndtn = 6 if cndtn == 2750
replace cndtn = 7 if cndtn == 3000
replace cndtn = 8 if cndtn == 4000
replace cndtn = 9 if cndtn == 5000
replace cndtn = 10 if cndtn == 6000
replace cndtn = 11 if cndtn == 7000
compress cndtn

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

label val cndtn cndtn

label def meta ///
    99 "Everything Else" ///
	1 "Collectibles" ///
	220 "Toys & Hobbies" ///
	237 "Dolls & Bears" ///
	260 "Stamps" ///
	267 "Books" ///
	281 "Jewelry & Watches" ///
	293 "Consumer Electronics" ///
	316 "Specialty Services" ///
	550 "Art" ///
	619 "Musical Instruments & Gear" ///
	625 "Cameras & Photo" ///
	870 "Pottery & Glass" ///
	888 "Sporting Goods" ///
	1249 "Video Games & Consoles" ///
	1281 "Pet Supplies" ///
	1305 "Tickets & Experiences" ///
	2984 "Baby" ///
	3252 "Travel" ///
	6000 "eBay Motors" ///
	10542 "Real Estate" ///
	11116 "Coins & Paper Money" ///
	11232 "DVDs & Movies" ///
	11233 "Music" ///
	11450 "Clothing Shoes & Accessories" ///
	11700 "Home & Garden" ///
	12576 "Business & Industrial" ///
	14339 "Crafts" ///
	15032 "Cell Phones & Accessories" ///
	20081 "Antiques" ///
	26395 "Health & Beauty" ///
	45100 "Entertainment Memorabilia" ///
	58058 "Computers/Tablets & Networking" ///
	64482 "Sports Mem Cards & Fan Shop" ///
	172008 "Gift Cards & Coupons", replace

label val meta meta