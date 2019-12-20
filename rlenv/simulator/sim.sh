type_passed=false
part_passed=false
SCRIPT_PATH=repo/rlenv/simulator/check_done.py

broken () {
    echo "[-t] or [-p] forgotten or invalid. Use [-h] for more information"
    exit 1;
}

usage () {
    echo "Description: Runs the market simulation to generate value estimates or discriminator inputs"
    echo -e "\n"
    echo "Arguments:"
    echo - e"\t -t: type of simulation -- one of [val, discrim]";
    echo -e "\t -p: partition of data -- one of [train_rl, train_models, test]";
    exit;
}

sim_done () {
  echo "The simulation has already been completed."
  echo "Please run rm done_* in ${PART}/${SIM_TYPE}/, then re-run this script to run the simulation"
  exit;
}

while getopts t:hp: option
do
    case "${option}" in
        t) SIM_TYPE=${OPTARG}; type_passed=true;;
        p) PART=${OPTARG}; part_passed=true;;
        h) usage; exit;;
        :) echo "Missing option argument for -$OPTARG. Use -h for usage instructions" >&2; exit 1;;
        *) echo "Unimplemented option: -$OPTARG" >&2; exit 1;;
    esac
done

if [ "$type_passed" = false ] || [ "$part_passed" = false ] 
then
    broken
fi

# get argument value from 
if [ "$SIM_TYPE" = "val" ]
then
    if ! python "${SCRIPT_PATH}" --part "${PART}" --values
    then
      sim_done
    fi
    qsub repo/rlenv/simulator/values/generate.sh "${PART}"

elif [ "$SIM_TYPE" = "discrim" ]
then
    if ! python "${SCRIPT_PATH}" --part "${PART}"
    then
      sim_done
    fi
    qsub repo/rlenv/simulator/discrim/generate.sh "${PART}"
else
    broken
fi
