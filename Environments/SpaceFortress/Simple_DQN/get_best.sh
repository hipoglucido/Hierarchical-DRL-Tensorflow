game="$1"
experiment="$2"

# csv file with results
csv=runs/$game/$experiment/${game}_${experiment}.csv

# the directory where the weights are stored
weights_dir=runs/$game/$experiment/weights

# get the epoch with the best reward
epoch=`cat $csv | grep test | tr "," " "| sort -nrk 5 | head -1 | cut -d ' ' -f1`

cp ${weights_dir}/${game}_${experiment}_${epoch}.prm snapshots/${game}_${experiment}_Best.prm