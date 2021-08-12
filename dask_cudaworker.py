import os
import time
import asyncio
from dask_cuda import CUDAWorker

def run_as_cuda_worker(scheduler_file):
    """
    Launch CUDAWorkers
    """
    async def _run_cuda_worker(scheduler_file):
        await CUDAWorker(scheduler_file=scheduler_file)

    task_rank = int(os.getenv("SLURM_PROCID"))
    nb_tasks = int(os.getenv("SLURM_NTASKS"))
    print("<< Dask CUDAWorker is being launched at rank {} / {} >>".format(task_rank, nb_tasks - 1))

    loop = asyncio.get_event_loop()
    asyncio.ensure_future(_run_cuda_worker(scheduler_file))
    loop.run_forever()
    loop.close()