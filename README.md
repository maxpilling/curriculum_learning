# Reinforcement Learning For Games

## Install Instructions

### Singularity

The way used throughout the project was via Singularity.

* Install Singularity - https://singularity.lbl.gov/install-linux
  * This must be done on a personal machine! Singularity needs `root` access for the initial build.
* Swap to the repo folder.
* Run `sudo singularity build starcraft.simg Singularity`.
  * This is going to install everything, so takes a while.
* Can be used either with:
  * `singularity shell -C starcraft.simg`
  * `singularity exec starcraft.simg python run.py ${SCRIPT_ARGS}`
  * If running on a machine that has a GPU + CUDA, ensure to pass `--nv` after `shell` or `exec`.

### Virtual Env

Only used initially, so may not be fully working.

* Run the `setup.sh` script, which will setup a Python virtual environment, and install SC2 if needed.
* Run `. ./start.sh` to enter the Python virtual environment.
* Run `deactivate` to leave once finished.

## Instructions to Run

An example script of running the CNN can be found in `CNN\runPySC2.sh`, which uses Singularity.
The paths in this script will need updating for a different user.
