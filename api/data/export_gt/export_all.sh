#!/bin/bash

# DDAD
echo "-> Exporting DDAD targets"
python api/data/export_gt/ddad.py --mode val --save-stem targets_val

# DIODE
echo "-> Exporting Diode 'indoor' targets"
python api/data/export_gt/diode.py --mode val --scene indoors --save-stem targets_val_indoors

echo "-> Exporting Diode 'outdoor' targets"
python api/data/export_gt/diode.py --mode val --scene outdoor --save-stem targets_val_outdoor

# KITTI
echo "-> Exporting Kitti Eigen targets"
python api/data/export_gt/kitti.py --split eigen --mode test --use-velo-depth 1 --save-stem targets_test

echo "-> Exporting Kitti Eigen Zhou targets"
python api/data/export_gt/kitti.py --split eigen_zhou --mode test --use-velo-depth 0 --save-stem targets_test

echo "-> Exporting Kitti Eigen Benchmark targets"
python api/data/export_gt/kitti.py --split eigen_benchmark --mode test --use-velo-depth 0 --save-stem targets_test

# MANNEQUIN CHALLENGE
echo "-> Exporting Mannequin Challenge targets"
python api/data/export_gt/mannequin.py --mode test --save-stem targets_test

# NYUD
echo "-> Exporting NYUD targets"
python api/data/export_gt/nyud.py --mode test --save-stem targets_test

# SINTEL
echo "-> Exporting Sintel targets"
python api/data/export_gt/sintel.py --mode train --save-stem targets_train

# TUM
echo "-> Exporting TUM targets"
python api/data/export_gt/tum.py --mode test --save-stem targets_test
