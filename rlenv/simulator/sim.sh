type_passed=false

broken () {
    echo "-t and -p must be given as an argument. Use -h for more information"
    exit 1;
}

usage () {
    echo "This script runs the market simulation to generate value estimates or discriminator inputs"
    echo "Arguments:"
    echo "\t -t: type of simulation -- one of [val, discrim]";
    echo "\t -p: partition of data -- one of [train_rl, train_models, test]"
    exit;
}

while getopts t:h option
do
case "${option}"
in
t) SIM_TYPE=${OPTARG}; ;;
p) PART
h) usage; exit;;
:) echo "Missing option argument for -$OPTARG. Use -h for usage instructions" >&2; exit 1;s;
*) echo "Unimplemented option: -$OPTARG" >&2; exit 1;;


if [ ! $type_passed ] || [ ! $part_passed ] 
then
    broken()
fi

# get argument value from 
if [ "$SIM_TYPE" = "val" ]
then
    SIM_ARG="--values";
elif [ "$SIM_TYPE" = "discrim" ]
then
    SIM_ARG="";
else
    broken()
fi

if [! repo/rlenv/simulator/check_done.py --part ${PART} $SIM_ARG ]
then
    echo "The simulation has already been completed."
    echo "Please delete the files in ${PART}/done/${SIM_TYPE}/ and re-run this script"
fi

