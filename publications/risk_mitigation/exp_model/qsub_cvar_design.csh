#! /bin/csh
#$ -j y -o $JOB_NAME.$JOB_ID.log
#$ -cwd
#$ -m e -M kpk15@ic.ac.uk

echo "Running on $HOST with $NSLOTS core(s) allocated"

module load anaconda3/2018.12 
setenv PATH ~/.conda/envs/pydexenv_yml/bin:$PATH
python ./cvar_design.py
