from typing import List, Optional
import asyncio

from .job import Job, Command
from .worker import Worker


class WorkHouse:
    def __init__(self, num_workers: Optional[int]=None, worker_ids: Optional[List[int]]=None, jobs: List[Job]=None):
        assert (num_workers is not None) or (worker_ids is not None)
        self.jobs = jobs
        self.worker_ids = worker_ids if worker_ids is not None else [i for i in range(num_workers)]
        self.workers = None
        self.next_job_idx = 0

    async def spawn_workers(self):
        self.workers = [Worker(id=i, next_job=self.get_next_job) for i in self.worker_ids]
        await asyncio.wait([w.start() for w in self.workers])

    def get_next_job(self) -> Optional[Job]:
        if self.next_job_idx == len(self.jobs):
            return None
        else:
            j = self.jobs[self.next_job_idx]
            self.next_job_idx += 1
            print(f"Progess: [{self.next_job_idx} / {len(self.jobs)}]")
            return j
