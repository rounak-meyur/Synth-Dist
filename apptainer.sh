#!/bin/bash

module load apptainer
module load python/3.11.5

export APPTAINER_TMPDIR=/scratch/$USER/APPTAINER
export APPTAINER_CACHEDIR=/scratch/$USER/APPTAINER

rm -rf $APPTAINER_TMPDIR
rm -rf $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR
mkdir -p $APPTAINER_CACHEDIR


apptainer build /qfs/projects/cameo/apptainer/synthdist_1.0.sif apptainer.def