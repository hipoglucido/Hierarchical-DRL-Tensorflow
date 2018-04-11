game=$1
experiment=$2
version=-v0
backend=$3

if [ -z $3 ]; then
	backend=gpu
elif [ $3 == "CPU" -o $3 == "cpu" ]; then
	backend=cpu
else
	backend=gpu
fi

weights=snapshots/${game}_${experiment}_Best.prm

# get snapshot if not present
if [ ! -f ${weights} ]; then
	./get_best.sh $game $experiment $backend
fi

python src/main.py  ${game}${version}\
			--backend ${backend}\
			--environment gym\
			--play_games 20\
			--display_screen true\
			--load_weights ${weights}
