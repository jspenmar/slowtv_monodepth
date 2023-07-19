#!/bin/bash

file=DDAD.tar
wget -c https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/${file}
tar -xvf ${file} && rm ${file}
