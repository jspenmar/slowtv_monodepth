#!/bin/bash

for file in train val test; do
    wget -c https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/dataset/${file}.zip
    unzip ${file}.zip && rm ${file}.zip
done