<div align="center">

# Model-Based State-Diffuser for Sample Efficient Online Reinforcement Learning

Authors: Soohyuk Cho, Dun-Ming Huang, Ryan Liu

</div>

Welcome to code repository for Model-Based State-Diffuser for Sample Efficient Online Reinforcement Learning

The implementation of diffusion is based on [labml.ai](https://nn.labml.ai/?_gl=1*1c7ldgt*_ga*MTAwODcxMjU4Ny4xNjk0OTk1ODQy*_ga_PDCL9PHMHT*MTcwMjUwNTU1OS4xLjAuMTcwMjUwNTU1OS4wLjAuMA..) and [SynthER](https://github.com/conglu1997/SynthER/blob/main/synther/diffusion/elucidated_diffusion.py). The infrastructe and SAC agent is modified from 
[UC Berkeley CS 285 HW5](https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/homeworks/hw5.pdf).

## Installation 
```
git clone git@github.com:ryanliu30/StateDiffusion.git
cd StateDiffusion
conda create -n StateDiffusion python=3.10
conda activate StateDiffusion
pip install -r requirements.txt
pip install -e .
```
## Usage
To begin with, run the following command:
```
python cs285/scripts/run.py --dynamics_config_file experiments/dynamics/halfcheetah.yaml --sac_config_file experiments/sac/vanilla.yaml
```
This will run the baseline vanilla SAC in the halfcheetah envoronment.
For MBPO:
```
python cs285/scripts/run.py --dynamics_config_file experiments/dynamics/halfcheetah.yaml --sac_config_file experiments/sac/mbpo.yaml
```
For SynthER:
```
python cs285/scripts/run.py --dynamics_config_file experiments/dynamics/halfcheetah.yaml --sac_config_file experiments/sac/mbpo.yaml --diffusion_config experiments/diff/all.yaml
```
For State Diffusion:
```
python cs285/scripts/run.py --dynamics_config_file experiments/dynamics/halfcheetah.yaml --sac_config_file experiments/sac/mbpo.yaml --diffusion_config experiments/diff/state.yaml
```

## Reproduction
To reproduce our result, we recommend to run the following command:
```
python run_all.py --name halfcheetah;
python run_all.py --name hopper;
python run_all.py --name walker;
```
To produce the plots, please run the `make_plots.ipynb` notebook. The logs we used to write the report is also provided in the repo.
