#!/usr/bin/env bash

game="$1"
experiment="$2"
# standard version
version=-v0
backend=gpu

if [ -z $3 ]; then
	backend=gpu
elif [ $3 == "CPU" -o $3 == "cpu" ]; then
	backend=cpu
fi

# the directory where the weights are stored
weights_dir=runs/$game/$experiment/weights

# the name of the last weights we managed to reach
weights_name=`ls -v $weights_dir | tail -n 1`

# the last epoch we managed to reach
epoch=`echo $weights_name | cut -d . -f 1 | cut -d _ -f 3`

# if there is not yet an epoch reached, exit
if [ -z "$epoch" ]; then
	echo "No weights file not found."
	echo "Check if the .prm file is in the correct path."
fi

# the location of the weights were using
get_weights=$weights_dir/$weights_name

# the prefix of the name where the weights will be stored
store_weights=$weights_dir/${game}_${experiment}

# place where the results will be stored as csv
results=runs/$game/$experiment/${game}_${experiment}.csv

# combine all arguments in the python call
python src/main.py 	${game}${version}\
			--environment gym \
			--display_screen rgb_array \
			--save_weights_prefix ${store_weights}\
			--csv_file ${results}\
			--load_weights ${get_weights}\
			--start_epoch ${epoch}
