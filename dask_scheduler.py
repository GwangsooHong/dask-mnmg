import asyncio
import os
import json
import sys
from dask.distributed import Scheduler

def run_as_scheduler(scheduler_file):
    """
    Launch Dask Scheduler
    """
    async def _run_scheduler():
        async with Scheduler() as s:
            with open(scheduler_file, "w") as f:
                json.dump(s.identity(), f, indent=2)
            await s.finished()
    
    task_rank = int(os.getenv("SLURM_PROCID"))
    nb_tasks = int(os.getenv("SLURM_NTASKS"))

    if task_rank == 1:
        print("<< Dask Scheduler is being launched at rank {} / {} >>".format(task_rank, nb_tasks - 1))

        loop = asyncio.get_event_loop()
        loop.run_until_complete(_run_scheduler())    
        loop.close()
    else:
        sys.exit(0)

    