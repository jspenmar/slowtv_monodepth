#!/bin/bash

ROOT=${1}

echo "-> Exporting Kitti Eigen-Zhou depth hints"
python api/data/postprocess/compute_kitti_hints.py --split eigen_zhou --mode train --root ${ROOT}
python api/data/postprocess/compute_kitti_hints.py --split eigen_zhou --mode val --root ${ROOT}

echo "-> Exporting Kitti Eigen depth hints"
python api/data/postprocess/compute_kitti_hints.py --split eigen --mode train --root ${ROOT}
python api/data/postprocess/compute_kitti_hints.py --split eigen --mode val --root ${ROOT}

echo "-> Exporting Kitti Eigen-Benchmark depth hints"
python api/data/postprocess/compute_kitti_hints.py --split eigen_benchmark --mode train --root ${ROOT}
python api/data/postprocess/compute_kitti_hints.py --split eigen_benchmark --mode val --root ${ROOT}
