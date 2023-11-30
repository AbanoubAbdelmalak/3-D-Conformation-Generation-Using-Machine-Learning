#!/bin/bash
#SBATCH --output=slurm.%j.out    # STDOUT
#SBATCH --error=slurm.%j.err     # STDERR
#SBATCH --partition=any          # partition (queue)
#SBATCH --ntasks=1               # use 1 task
#SBATCH --mem=100                # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=2:00              # total runtime of job allocation ((format D-HH:MM:SS; first parts optional)

# start program
python3 test.py
