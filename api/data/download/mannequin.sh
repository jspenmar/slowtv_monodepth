#!/bin/bash

file=MannequinChallenge.tar
wget -c https://storage.googleapis.com/mannequinchallenge-data/${file}
tar -xvf ${file} && rm ${file}

mv MannequinChallenge/* . && rmdir MannequinChallenge

# TODO: This only contains info about the splits, not the images/depths themselves