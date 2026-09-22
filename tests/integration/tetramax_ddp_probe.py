"""Small two-rank CPU/Gloo check; no model loading or licensed processes.

Run with PYTHONPATH=atpgllm:data_preprocessing python this_file.py.
"""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, rendezvous):
    from atpgllm.training.tools import ToolScheduler, abort_on_simulator_failure
    from tetramax_seats import SimulationError
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    class Accelerator:
        device=torch.device('cpu')
        num_processes=2
        def gather(self, tensor):
            result=[torch.empty_like(tensor) for _ in range(2)]
            dist.all_gather(result,tensor)
            return torch.cat(result)
    trainer=SimpleNamespace(accelerator=Accelerator())
    os.environ['TMAX_PIPELINED_TOOLS']='1'
    prompt='Target "sa0 y" '+repr({'doc_id':'doc','netlist':'module design(input a,output y); endmodule'})
    call={'name':'fault_simulation_tool','arguments':{'input_vector':{'a':1},'output_vector':{'y':1},'fault':'sa0 y','doc_id':'doc'}}
    try:
        for inject_failure in (False,True):
            def handler(**kwargs):
                time.sleep(0.05 if rank==0 else 0.4)
                if inject_failure and rank==1:
                    raise SimulationError('injected simulator failure')
                return str(rank)
            count=0
            stopped=False
            with ToolScheduler({'fault_simulation_tool':handler}) as scheduler:
                calls,indices=[call],[0]
                try:
                    while True:
                        calls,indices,outcomes=scheduler.take(trainer,calls,indices,[prompt])
                        abort_on_simulator_failure(trainer,any(failed for _,failed in outcomes))
                        gen=trainer.accelerator.gather(torch.tensor([bool(indices)]))
                        if not gen.any().item():
                            pending=trainer.accelerator.gather(torch.tensor([bool(scheduler.pending)]))
                            if pending.any().item():
                                calls,indices=[],[]
                                continue
                            break
                        # Stand-in for the collective generation call.
                        count+=len(indices)
                        dist.barrier()
                        calls,indices=[],[]
                except SimulationError:
                    stopped=True
            assert stopped==inject_failure
            if not inject_failure:
                assert count==1
            dist.barrier()
        if rank==0:
            print('Two-rank scheduler and coordinated failure check passed',flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='tmax_ddp_') as directory:
        mp.spawn(worker,args=(str(Path(directory)/'rendezvous'),),nprocs=2,join=True)
