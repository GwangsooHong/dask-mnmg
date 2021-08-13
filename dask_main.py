#!/opt/conda/envs/rapids/bin/python

import os
import sys
import subprocess
import argparse
import glob
import time
import numpy as np

import dask_cudf
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from cuml.dask.cluster.kmeans import KMeans as cuKMeans
from cuml.dask.cluster.dbscan import DBSCAN as cuDBSCAN

parser = argparse.ArgumentParser(description="Arguemnts for testing RAPIDS Dask")
parser.add_argument('--slurm', default=False, help="slurm scheduler if True else Local Cluster")
parser.add_argument("--download-criteo", default=False, help="Download CRITEO dataset if True")
parser.add_argument("--number-days", default=2, help="Required only if --download-criteo is True. Specify how many days you want to download")
parser.add_argument("--decompress-criteo", default=False, help="Decompress gz format if True. You should have downloaded gz format dataset")
parser.add_argument("--parquet-prepared", default=True, help="Set True if you have already finished CRITEO parquet format conversion")
parser.add_argument("--base-dir", default=os.getcwd(), help="base workspace")
parser.add_argument("--gpus-per-node", default=4, help="number of gpus per node required to Dask Local Cluster. You don't need to specify when using Slurm")
parser.add_argument("--clustering-algorithm", default="KMeans", help="KMeans is the only one supported on Multi-nodes")
args = parser.parse_args()

SLURM = args.slurm

DOWNLOAD_CRITEO = args.download_criteo
NUMBER_DAYS = args.number_days
DECOMPRESS_CRITEO = args.decompress_criteo
PARQUET_PREPARED = args.parquet_prepared

BASE_DIR = args.base_dir
ORG_DATA_PATH = os.path.join(BASE_DIR, "crit_orig")
PARQUET_DATA_PATH = os.path.join(BASE_DIR, "crit_parquet")
DASK_WORK_PATH = os.path.join(BASE_DIR, "dask-worker-space")
    
GPUS_PER_NODE = int(subprocess.check_output("nvidia-smi -L|wc -l", shell=True)) if SLURM else int(args.gpus_per_node)
CLUSTERING_ALGORITHM = args.clustering_algorithm

def download_dataset():
    '''
    Download CRITEO dataset.
    Refer to https://nvidia.github.io/NVTabular/v0.5.3/examples/scaling-criteo/01-Download-Convert.html
    '''

    from nvtabular.utils import download_file

    # Test if NUMBER_DAYS in valid range
    if NUMBER_DAYS < 2 or NUMBER_DAYS > 23:
        raise ValueError(
            str(NUMBER_DAYS)
            + " is not supported. A minimum of 2 days are "
            + "required and a maximum of 24 (0-23 days) are available"
        )

    # Create BASE_DIR if not exists
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    # Create input dir if not exists
    if not os.path.exists(ORG_DATA_PATH):
        os.makedirs(ORG_DATA_PATH)

    # Iterate over days
    for i in range(0, NUMBER_DAYS):
        file = os.path.join(ORG_DATA_PATH, "day_" + str(i) + ".gz")
        # Download file, if there is no .gz, .csv or .parquet file
        if not (
            os.path.exists(file)
            or os.path.exists(
                file.replace(".gz", ".parquet").replace("crit_orig", "crit_parquet")
            )
            or os.path.exists(file.replace(".gz", ""))
        ):
            # cmd = "wget http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_" + str(i) + ".gz"
            # subprocess.call(cmd, shell=True)
            download_file(
                "http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_"
                + str(i)
                + ".gz",
                file,
            )
        print(file + "is downloaded.\n")


def decompress_gzip():
    '''
    Decompress gz format. 
    gz format doesn't support breaking apart files as chunk-size.
    Each individual file should be fit in memory if you don't decompress.
    '''

    # check if gz file exists in your org_data_path
    files = glob.glob(os.path.join(ORG_DATA_PATH, "*.gz"))
    if len(files) > 0:
        print("Decompressing {} ...".format(files))
        cmd = 'find {} -name "*.gz"|xargs gzip -d'.format(ORG_DATA_PATH)
        subprocess.call(cmd, shell=True)
        print("Decompressing is done")
        print(glob.glob(os.path.join(ORG_DATA_PATH, "*")))
        
    else:
        print("Nothing to decompress")
    

def get_criteo_meta():
    '''
    CRITEO dataset columns and dtypes info
    '''

    columns = {
            "continuous_columns": ["I" + str(x) for x in range(1, 14)],
            "categorical_columns" : ["C" + str(x) for x in range(1, 27)],
            "label" : ["label"]
    }
    dtypes = {}
    dtypes["label"] = np.int32
    for x in columns["continuous_columns"]:
        dtypes[x] = np.int32
    for x in columns["categorical_columns"]:
        # Note that "hex" means that the values will be hexadecimal strings that should be converted to int32
        dtypes[x] = "hex"

    criteo_meta = {}
    criteo_meta["columns"] = columns
    criteo_meta["dtypes"] = dtypes

    return criteo_meta


def convert_from_csv_to_parquet(client, part_mem_frac, meta):
    '''
    Convert criteo dataset into parquet format with some preprocessing included.
    '''
    
    parquet_list = glob.glob(os.path.join(PARQUET_DATA_PATH, "day_*.parquet"))
    if len(parquet_list) > 0:
        print("It seems your original dataset has been already converted")
        print("Parquet converting is passed")
    else:
        import nvtabular as nvt
        from nvtabular.ops import Normalize, FillMissing, Categorify, Clip, LogOp

        file_list = glob.glob(os.path.join(ORG_DATA_PATH, "day_*"))
        print("files need to be converted:", file_list)

        # Specify column names    
        continuous_columns = meta["columns"]["continuous_columns"]
        categorical_columns = meta["columns"]["categorical_columns"]
        label = meta["columns"]["label"]
        columns = label + continuous_columns + categorical_columns

        # Specify column dtypes
        dtypes = meta['dtypes']

        # NvTabular dataset
        dataset = nvt.Dataset(
            file_list,
            engine="csv",
            names=columns,
            part_mem_fraction=part_mem_frac,
            sep="\t",
            dtypes=dtypes
        )

        # Specify basic data preprocessing. Filling NA, Log-norm for continuous columns, freq_thresholding for categorical columns 
        cont_features = continuous_columns >> FillMissing() >> Clip(min_value=0) >> LogOp() >> Normalize()
        cat_features = categorical_columns #>> Categorify(freq_threshold=15) # ValueError: not enough values to unpack(expected 2, got 0)
        features = cat_features + cont_features + label

        # Applying preprocessing data and persist it as a parquet format
        workflow = nvt.Workflow(features, client=client)
        # print("calculating statistics for preprocessing...")
        # nvtd = workflow.fit(dataset)
        print("Start Parquet converting with preprocessing...")
        workflow.fit_transform(dataset).to_parquet(PARQUET_DATA_PATH, preserve_files=True)
        print("Parquet converting is finished")


def launch_dask_slurm_cluster():
    '''
    Launch Slurm Dask Client
    '''

    # Slurm parameters
    nb_tasks = int(os.getenv("SLURM_NTASKS"))
    task_rank = int(os.getenv("SLURM_PROCID"))
    job_id = int(os.getenv("SLURM_JOB_ID"))
    scheduler_file = "dask-scheduler-{}.json".format(job_id)
    gpus_per_node = GPUS_PER_NODE
    total_num_workers = gpus_per_node * nb_tasks - 2 * gpus_per_node

    # Connect dask Client at rank 0
    if task_rank == 0:
        print("<< Dask-Client is launched at rank {} >>".format(task_rank))
        client = Client(scheduler_file=scheduler_file, 
                        local_directory = DASK_WORK_PATH)
        workers = client.ncores()

        while len(workers) < total_num_workers:
            print("waiting for workers {} / {}".format(len(workers), total_num_workers))
            time.sleep(1)
            workers = client.ncores()
    
        print("all workers are finally available {} / {}".format(len(workers), total_num_workers))               
        return client
    else:
        pass


def launch_dask_local_cluster(device_limit_frac, device_pool_frac, nGPUs=2, managed_memory=False):
    '''
    Launch local Dask Client.
    '''

    import pynvml
    pynvml.nvmlInit()
    
    # Device parameters
    device_size = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).total) #memory size per gpu
    device_limit = int(device_limit_frac * device_size)       # Spill GPU-Worker memory to host at this limit.
    device_pool_size = int(device_pool_frac * device_size)    # Memory pool size

    # Connect Dask Client
    cmd = "hostname --all-ip-addresses"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    _ipaddr = str(output.decode()).split()[0]

    cluster = None 
    if cluster is None:
        cluster = LocalCUDACluster(
            n_workers = int(nGPUs),
            enable_tcp_over_ucx = True,
            enable_nvlink = True,
            rmm_managed_memory = managed_memory,
            rmm_pool_size = device_pool_size,
            device_memory_limit = device_limit,
            local_directory = DASK_WORK_PATH,
            ip=_ipaddr)
    client = Client(cluster)    
    return client


def run_kmeans(ddf, init="k-means||", n_clusters = 8, random_state = 100):
    '''
    Run Dask KMeans for multi-node multi-gpus
    '''
    
    print("- Parameters")
    print("     - init:", init)
    print("     - n_clusters:", n_clusters)
    print("     - random_state:", random_state)
    start = time.time()
    kmeans_cuml = cuKMeans(init=init,
                       n_clusters=n_clusters,
                       verbose=5,
                       random_state=random_state)
    kmeans_cuml.fit(ddf)
    elapsed_time = time.time() - start
    print("- KMeans elapsed time:", elapsed_time, "seconds\n")
    return kmeans_cuml, elapsed_time


def run_dbscan(ddf, eps = 0.5, min_samples = 5):
    '''
    Run Dask DBSCAN for multi-node multi-gpus
    '''

    print("- Parameters")
    print("     - eps:", eps)
    print("     - min_samples:", min_samples)
    start = time.time()
    dbscan_cuml = cuDBSCAN(eps = eps,
                           min_samples = min_samples,
                           verbose = 5)
    dbscan_cuml.fit(ddf)
    elapsed_time = time.time() - start
    print("- DBSCAN elapsed time:", elapsed_time, "seconds\n")
    return dbscan_cuml, elapsed_time


def main():  
    if CLUSTERING_ALGORITHM != "KMeans" and CLUSTERING_ALGORITHM != "DBSCAN":
        print("clustering algorithm must be one of 'KMeans' or 'DBSCAN'")
        sys.exit(1)
    
    if DOWNLOAD_CRITEO:
        # Download Criteo dataset if needed"
        download_dataset()
        print("download complete")
        sys.exit(0)

    if DECOMPRESS_CRITEO:
        # gz format decompression (because gz doesn't support chunk-reading in dataframe)
        decompress_gzip()
        print("decompress complete")
        sys.exit(0)
    
    if SLURM:
        # Launch DASK Slurm GPU Cluster
        client = launch_dask_slurm_cluster()

    else:
        # Launch Dask Local GPU Cluster
        client = launch_dask_local_cluster(device_limit_frac = 0.5, device_pool_frac = 0.9, nGPUs = GPUS_PER_NODE)

    if client is None:
        sys.exit(0)    

    print("<< Dask Cluster is successfully launched >>")
    print("- Dashboard link:", client.dashboard_link)
    print("- Base Directory:", BASE_DIR)
    print("- Parquet Data Directory", PARQUET_DATA_PATH)
    print("- Slurm enabled:", SLURM)
    if SLURM:
        print("- Slurm job id:", int(os.getenv("SLURM_JOB_ID")))
        print("- Num Dask Workers:", int(os.getenv("SLURM_NTASKS")) - 2) 
    print("- Num GPUs per Worker:", GPUS_PER_NODE)
    print("- Total Num GPUs:", len(client.ncores()))
    print("- Clustering algorithm:",  CLUSTERING_ALGORITHM)
    
    # Get Parquet format
    criteo_meta = get_criteo_meta()
    columns = criteo_meta["columns"]["label"] +  criteo_meta["columns"]["continuous_columns"] + criteo_meta["columns"]["categorical_columns"]
    
    # Parquet is a compressed, column-oriented file structure and require less disk space.
    if PARQUET_PREPARED != True:
        convert_from_csv_to_parquet(client = client, part_mem_frac = 0.1, meta = criteo_meta)
    parquet_file_list = glob.glob(os.path.join(PARQUET_DATA_PATH, "*"))
    print("- Num files to be loaded:", len(parquet_file_list),"\n")

    ddf = dask_cudf.read_parquet(path = parquet_file_list,
                                 columns = columns,
                                 split_row_groups=True)
    continuous_features = criteo_meta["columns"]["continuous_columns"]
    test_ddf = ddf[continuous_features]
 
    print("<< Persisting data... >>")
    stime = time.time()    
    test_ddf = test_ddf.persist()
    
    if SLURM:
        future = client.submit(len, test_ddf)
        print("- number of rows:", future.result())
    else:
        print("- number of rows:", len(test_ddf))
    print("- number of features:", len(continuous_features))
    print("- number of dask partitions:", test_ddf.npartitions)
    print("- Total data size:", subprocess.check_output(['du','-sh', PARQUET_DATA_PATH]).split()[0].decode('utf-8'))
    elapsed_time_persisting = time.time() - stime
    print("- Persisting elapsed time:", elapsed_time_persisting, "seconds\n")
    
    if CLUSTERING_ALGORITHM == "KMeans":
        print("<< KMeans computing ... >>")
        output, elapsed_time = run_kmeans(test_ddf)
    else:
        # dask_cudf is not supported. DBSCAN should fit your data on single GPU memory size and replicates it to other GPUs for computing.
        print("<< DBSCAN computing ... >>")
        print("- Fitting data to your single GPU memory")
        test_ddf = test_ddf.compute()        
        output, elapsed_time = run_dbscan(test_ddf)
    
    client.close()
    
    if SLURM:
        # remove scheduler file
        os.remove("dask-scheduler-" + str(os.getenv("SLURM_JOB_ID")) + ".json")
        # remove garbages
        subprocess.call("rm -rf cudf_cufile_config.* ", shell=True)
    print("<< Dask job successfully is finished. >>")
    print("- Shutting down client ...")
    time.sleep(5)
    client.shutdown()
    print("- Dask client is closed.\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
