from typing import List, Callable, Optional
import asyncio

from .job import Job, Command


class Worker:
    def __init__(self, id: int, next_job: Callable[[], Optional[Job]]):
        self.id = id
        self.next_job = next_job

    def log(self, s: str):
        s = f'worker {self.id} says: {s}'
        print(s)

    async def start(self):
        while True:
            job: Optional[Job] = self.next_job()
            # No job any more!
            # My work has been finished. :)
            if job is None:
                self.log('exiting')
                return

            self.log('running a new job')
            res = await job.run(self.id)
            self.log(f'job results: {res}')

            # So tired! Let's rest for one second.
            self.log('sleeping')
            await asyncio.sleep(1)
