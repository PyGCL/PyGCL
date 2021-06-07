from typing import List, Optional, Callable
import asyncio


class Command:
    def __init__(self, exe: str, args: List[str], stdout_redirect: Optional[str] = None):
        self.exe = exe
        self.args = args
        self.stdout_redirect = stdout_redirect

    def cmd(self):
        res = f'{self.exe} {" ".join(self.args)}'
        if self.stdout_redirect is not None:
            res += f' >> {self.stdout_redirect}'
        return res

    async def execute(self):
        print(f'running command: {self.cmd()}')
        proc = await asyncio.create_subprocess_exec("sh", "-c", self.cmd(), stdout=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        stdout = stdout.decode() if stdout else '[EMPTY]'
        stderr = stderr.decode() if stderr else '[EMPTY]'
        return stdout, stderr


class Job:
    def __init__(self):
        pass

    def resolve(self, worker_id: int) -> List[Command]:
        raise NotImplementedError

    async def run(self, worker_id):
        cmds = self.resolve(worker_id)
        res = []
        for cmd in cmds:
            r = await cmd.execute()
            res.append(r)
        return res

    async def __call__(self, worker_id: int):
        await self.run(worker_id)
