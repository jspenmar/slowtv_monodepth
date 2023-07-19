#!/bin/bash

fr2=(
freiburg2_desk_with_person
)

fr3=(
freiburg3_sitting_static
freiburg3_sitting_xyz
freiburg3_sitting_halfsphere
freiburg3_sitting_rpy
freiburg3_walking_static
freiburg3_walking_xyz
freiburg3_walking_halfsphere
freiburg3_walking_rpy
)

download_fr2_scene () {
  local cat=$1
  echo $cat

  file=rgbd_dataset_${cat}.tgz

  echo "Downloading: "$file
  wget https://vision.in.tum.de/rgbd/dataset/freiburg2/${file} && tar -xvzf ${file} && rm ${file}
}

download_fr3_scene () {
  local cat=$1
  echo $cat

  file=rgbd_dataset_${cat}.tgz

  echo "Downloading: "$file
  wget https://vision.in.tum.de/rgbd/dataset/freiburg3/${file} && tar -xvzf ${file} && rm ${file}
}


process_max=5
p_count=0
for i in ${fr2[@]}; do
    download_fr2_scene $i &
    if (( ++p_count == $process_max )); then
        wait
        p_count=0
    fi
done


p_count=0
for i in ${fr3[@]}; do
    download_fr3_scene $i &
    if (( ++p_count == $process_max )); then
        wait
        p_count=0
    fi
done