#!/bin/bash

file=MPI-Sintel-complete.zip
wget -c http://files.is.tue.mpg.de/sintel/${file}
unzip ${file} && rm ${file}

file=MPI-Sintel-depth-training-20150305.zip
wget -c http://files.is.tue.mpg.de/jwulff/sintel/${file}
unzip ${file} && rm ${file}

mv training train
