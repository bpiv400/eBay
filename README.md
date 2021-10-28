# eBay-processing
Replication files for "Optimal Bargaining on eBay".

Order of operations:
1. Clone this repo.
2. Create a virtual environment.
3. Install <code>rlpyt</code> as an editable package. See [here](https://github.com/astooke/rlpyt).
4. Install the packages in <code>requirements.txt</code>.
5. Change the constants <code>DATA_DIR</code> and <code>FIG_DIR</code> in <code>paths.py</code> to point to the folders where the data and figures, respectively, will reside.
6. Download the data from [here](https://www.dropbox.com/s/bntba3fd0miissy/eBay.7z?dl=0) and unzip it into your data folder.
   1. If you would like to generate the input features yourself, see this [repo](https://github.com/etangreen/eBay-processing).
7. Run the bash scripts in <code>bash/</code>.