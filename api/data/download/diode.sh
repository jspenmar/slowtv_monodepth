#!/bin/bash

file=data_list.zip
wget -c https://diode-1254389886.cos.ap-hongkong.myqcloud.com/${file}
unzip ${file} && rm ${file} && rm -rf __MACOSX

file=val.tar.gz
wget -c http://diode-dataset.s3.amazonaws.com/${file}
tar -xvf ${file} && rm ${file}
