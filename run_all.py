import multiprocessing
import subprocess
from argparse import ArgumentParser

def work(cmd):
    return subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--name", "-n", type = str, required = True)
    args = parser.parse_args()
    pool = multiprocessing.Pool(processes=8)
    name = args.name
    tasks = sum([[
        f"python cs285/scripts/run.py --dynamics_config_file experiments/dynamics/{name}.yaml --sac_config_file experiments/sac/mbpo.yaml --seed {seed}",
        f"python cs285/scripts/run.py --dynamics_config_file experiments/dynamics/{name}.yaml --sac_config_file experiments/sac/vanilla.yaml --seed {seed}",
        f"python cs285/scripts/run.py --dynamics_config_file experiments/dynamics/{name}.yaml --sac_config_file experiments/sac/mbpo.yaml --diffusion_config experiments/diff/state.yaml --seed {seed}",
        f"python cs285/scripts/run.py --dynamics_config_file experiments/dynamics/{name}.yaml --sac_config_file experiments/sac/mbpo.yaml --diffusion_config experiments/diff/all.yaml --seed {seed}"
    ] for seed in range(3)], start = [])
    r = pool.map_async(work, tasks)
    r.wait() # Wait on the results