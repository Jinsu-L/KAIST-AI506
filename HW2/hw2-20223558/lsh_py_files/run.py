import os
import sys
import subprocess
from itertools import chain

def main():
    workspace = sys.argv[1]

    def run_with_args(_in="", _out="", jobname="Example", options={}, envs={}, num_reducers=5):
        try:
            return subprocess.run(["mapred", "streaming", "-D", f"mapreduce.job.reduces={num_reducers}",
                                   *chain.from_iterable([["-D", f"{k}={v}"] for k, v in options.items()]),
                                   "-file", f"./{jobname}Mapper.py", "-file", f"./{jobname}Reducer.py",
                                   "-input", workspace + _in, "-output", workspace + _out,
                                   "-mapper", f"./{jobname}Mapper.py", "-reducer", f"./{jobname}Reducer.py",
                                   *chain.from_iterable([["-cmdenv", f"{k}={v}"] for k, v in envs.items()])]).returncode
        except:
            return 1
    
    # if run_with_args("/inputpath", "/outputpath", "example", options={}, num_reducers=5) != 0: sys.exit(1)
    if run_with_args("/", "/subtask1", "Subtask1", options={}, num_reducers=5) != 0: sys.exit(1)
    if run_with_args("/subtask1", "/subtask2", "Subtask2", options={}, num_reducers=5) != 0: sys.exit(1)
    if run_with_args("/subtask2", "/output", "Subtask3", options={}, num_reducers=5) != 0: sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main()

