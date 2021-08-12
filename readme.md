# Run Rapids Dask with multi-gpus on multi-nodes

This implementation works based on [NGC(NVIDIA GPU Cloud)](https://ngc.nvidia.com/catalog/collections) RAPIDS container(rapidsai:21.08-cuda11.2-runtime-ubuntu18.04). `dask_main.py` is a main script file to run Dask on single or multi-nodes. It contains several arguments worth noting:


## 1. Download CRITEO dataset

CRITEO is one of the largest pulbic available dataset containing 24 files, each one has a size of around 15GB compressed gz format. The whole dataset contains around 1.3TB of uncompressed click logs consisting of 4 billion rows with 40 features(13 numerical figures, 26 categorical features, and 1 label). For more details, refer to https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/.

Download is available via `wget` command like
```bash
wget http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_${num}.gz
```
or if you are working inside RAPIDS container, for example, you can download the dataset from day 0 to day 3 as:
```bash
pip install -q nvtabular==0.6.1
python dask_main.py --download-criteo True --number-days 4 --base-dir ${PWD}
```

## 2. Decompress dataset
Original gz format doesn't support breaking apart files as chunk-size when loading on dataframe. Each individual file should be fit in memory if you don't decompress. 

You can use either linux gzip command directly,
```bash
gzip -d ${gzfilename}
```
or python implementation in RAPIDS container like 
```bash
python dask_main.py --decompress-criteo True
```
please check `ORG_DATA_PATH` variable in `dask_main.py` when using python implementation. 


## 3-1. Multi-gpus on single-node

Multi-gpus on single machine is implmented using `dask_cuda` `LocalCUDACluster`. If you execute `dask_main.py` with default arguments, it starts working based on `LocalCUDACluster`. Please make sure that CRITEO dataset has already been prepared with parquet format. Once CRITEO gz format is uncompressed, it becomes really huge(around 60GB per each). Parquet is a compressed, column-oriented file structure and requires less disk space. In addition, it works faster with RAPIDS cudf and nvtabular dataset. `dask_main.py` also implements this parquet conversion with some preprocessing such as NA handling, log-normalization, etc in RAPIDS container.

If parquet dataset is not prepared, you should add `--parquet-prepared False` flag when executing `dask_main.py` to go through this conversion. Once parquet is prepared, you can remove this flag for the next execution.

```bash
pip install -q nvtabular==0.6.1
python dask_main.py --parquet-prepared False
```
Once parquet is prepared, remove the flag when executing 

```bash
python dask_main.py
```

## 3-2. Multi-gpus on multi-nodes
Dask running on multi-nodes is implemented with Slurm. It requires at least three nodes(one for Dask-Client, one for Dask-Scheduler, and Dask-Workers) to run since those modules interfere with each other when launched on the same host. That means, **if you request N nodes to the Slurm cluster, Workers are only launched on N-2 nodes.** Please make sure that `srun` command arguments might depend on your Slurm configuration(gres option couldn't be supported). Here are some modifications you should go through.
1. Modify `${WORKSPACE}` in `srun_dask.sh`. 
2. Modify some `srun` command arguments in `srun_dask.sh` properly. For example, four-nodes with eight-gpus per each can be configured as
    ```bash
    srun -N 4 -G 32 -p batch --gpus-per-node=8 --job-name=computing-test --mpi=pmix --container-image=${CONTAINER_IMAGE}    --container-mounts ${PWD}:${WORKSPACE} bash -c "${WORKSPACE}/dask_multinode.sh"
    ```
    then, only two nodes(16 GPUs) are used for workers.
3. As mentioned in 3-1, if you don't have parquet dataset prepared, add `--parquet-prepared False` to the line which executes `dask_main.py` in `dask_multinode.sh` such as
    ```bash
    python -u dask_main.py --slurm True --base-dir ${WORKSPACE} --clustering-algorithm KMeans --parquet-prepared False
    ```

Once everything is well modified, execute `srun_dask.sh`
```bash
./srun_dask.sh
```