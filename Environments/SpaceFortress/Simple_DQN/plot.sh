game="$1"
experiment="$2"
inputfile=runs/$game/$experiment/${game}_${experiment}.csv

# ignore the extension of the output file
outputfile=`echo $2 | cut -d . -f 1`.png


if [ `wc -l < ${inputfile}` == 0 ];then
	echo "File not found"
	exit 1;
fi


if [ -z $3 ]; then
	python src/plot.py $inputfile
	exit
fi

python src/plot.py $inputfile --png_file $outputfile
