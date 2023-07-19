#! /bin/bash

MAPFREE_PY=/<PATH>/<TO>/<MINICONDA3>/mapfree/bin/python
MAPFREE_ROOT=/<PATH>/<TO>/map-free-reloc

function run() {
  model=${1}; seed=${2}; solver=${3}
  name=${model}_${seed}

  echo "Running ${name} with ${solver} solver."

  base_cfg=config/matching/mapfree/loftr_${solver}_dptkitti.yaml
  new_cfg=config/matching/mapfree/loftr_${solver}_${model}.yaml

  out_dir=results/loftr_${solver}_${model}/${seed}

  cp ${base_cfg} ${new_cfg} && sed -i -e "s/dptkitti/${name}/g" ${new_cfg}

  $MAPFREE_PY submission.py ${new_cfg} -o ${out_dir} --split val && rm ${new_cfg}
  $MAPFREE_PY -m benchmark.mapfree ${out_dir}/submission.zip --split val > ${out_dir}/metrics.json
}

cd $MAPFREE_ROOT || exit
PYTHONPATH_OLD=$PYTHONPATH
PYTHONPATH=$MAPFREE_ROOT

# Revert to old settings in case of failure.
trap "PYTHONPATH=$PYTHONPATH_OLD && cd -" EXIT

for MODEL in MiDaS DPT_Large DPT_BEiT_L_512; do
  for SEED in 042; do
    for SOLVER in pnp emat; do
      run midas_${MODEL} ${SEED} ${SOLVER}
    done
  done
done

for MODEL in indoor outdoor; do
  for SEED in 042; do
    for SOLVER in pnp emat; do
      run newcrfs_${MODEL} ${SEED} ${SOLVER}
    done
  done
done

for MODEL in garg hrdepth_MS; do
  for SEED in 042 195 335; do
    for SOLVER in pnp emat; do
      run benchmark_${MODEL} ${SEED} ${SOLVER}
    done
  done
done
