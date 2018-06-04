#! /bin/bash
#SBATCH --mem=64G -n 10
python main.py --mode=exp9 --paralel=6
