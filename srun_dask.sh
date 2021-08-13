#!/bin/bash
set -e

CONTAINER_IMAGE=nvcr.io#nvidia/rapidsai/rapidsai:21.06-cuda11.2-runtime-ubuntu18.04
WORKSPACE=/workspace
DASK_WORKSPACE=${PWD}/dask-worker-space
PYCACHE=${PWD}/__pycache__

if [ -d ${DASK_WORKSPACE} ] && echo "Previous Dask-worker-space still exists"
then
    rm -rf ${DASK_WORKSPACE}
    echo "${DASK_WORKSPACE} is removed"
fi
if [ -d ${PYCACHE} ] && echo "Previous __pycache__ still exists"
then
    rm -rf ${PYCACHE}
    echo "${PYCACHE} is removed"
fi

#Annex-B
#srun -N 4 -G 32 -p batch --gpus-per-node=8 --job-name=sa-computing-test --exclusive --overcommit --mpi=pmix --container-image=${CONTAINER_IMAGE} --container-mounts ${PWD}:${WORKSPACE} bash -c "${WORKSPACE}/dask_multinode.sh"

#Prometheus
srun -N 5 -p batch --job-name=computing-test --export ALL,OMPI_MCA_btl="^openib",OMPI_MCA_pml="ucx" --overcommit --container-image=${CONTAINER_IMAGE} --container-mounts ${PWD}:${WORKSPACE} bash -c "${WORKSPACE}/dask_multinode.sh"
