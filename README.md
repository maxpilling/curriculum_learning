# Reinforcement Learning For Games

## Install Instructions

### Singularity

The way used throughout the project was via Singularity.

*   Install Singularity - https://singularity.lbl.gov/install-linux
    *   This must be done on a personal machine! Singularity needs `root`
        access for the initial build.
*   Swap to the repo folder.
*   Run `sudo singularity build starcraft.simg Singularity.def`.
    *   This is going to install everything, so takes a while.
*   Can be used either with:
    *   `singularity shell -C starcraft.simg`
    *   `singularity exec starcraft.simg python run.py ${SCRIPT_ARGS}`
    *   If running on a machine that has a GPU + CUDA, ensure to pass `--nv`
        after `shell` or `exec`.

### Poetry

Only used initially, so may not be fully working.

* Install SC2. If installed to a non-standard location set the `$SC2PATH`
 environment variable to point to the install location.
    * An example of the install and setting this variable is in the
     `Singularity` install file.
    * Also install the maps, which are also listed in the `Singularity` file,
      alongside their password. They should be installed into `$SC2PATH/Maps`,
      where the maps folder may need making.
* Install [poetry](https://github.com/sdispater/poetry) with `pip install poetry`.
* Once installed, use `poetry install` inside this repo. This will create a
  virtualenv and install all needed packages into it.
* Then you can call the scripts like so:

```sh
# Basic example:
# Map names can be gotten from the map file names that you just downloaded.
poetry run python CNN/run.py --map_name MAP_NAME --model_name MODEL_NAME --training=False

# For example:
poetry run python CNN/run.py --map_name MoveToBeacon --model_name TestModel --training=True

# To load that model back at a later date and continue training:
poetry run python CNN/run.py --map_name MoveToBeacon --model_name TestModel --training=True --if_output_exists=continue

# To load that model back at a later date with no more training:
poetry run python CNN/run.py --map_name MoveToBeacon --model_name TestModel --training=False

# To use that model in a secondary training phase:
# This has not been tested, so its possible you'll need to instead run the initial model
# with curriculum_num=0, since I don't remember how that code works.
poetry run python CNN/run.py --map_name MoveToBeacon --model_name TestModel2 --training=True --curriculum_num=1 --previous_model=_files/models/TestModel
```

This way is meant for PC dev work, so the TensorFlow version does not need a GPU.
If this is needed, then call `poetry remove tensorflow` and `poetry add tensorflow-gpu`.

You can also enter the poetry shell like so

```sh
poetry shell
```

At that point, calling `python` will use the project venv.

## Instructions to Run

An example script of running the CNN can be found in `CNN\runPySC2.sh`, which
uses Singularity. The paths in this script will need updating for a different
user.

To run a specific pretrained CNN model, the script should be called as follows,
where `MAP_NAME` is the map or mini-game in question and `MODEL_NAME` is the
exact name of the model, as stored in `CNN\_files\`.

```sh
python run.py --map_name MAP_NAME --model_name MODEL_NAME --training=False
```

The folders for the models should look as follows:

```
CNN/_files/
├── models
│   └── test_model_20
│       ├── checkpoint
│       ├── model.ckpt-13000.data-00000-of-00001
│       ├── model.ckpt-13000.index
│       ├── model.ckpt-13000.meta
│       ├── model.ckpt-13500.data-00000-of-00001
│       ├── model.ckpt-13500.index
│       └── model.ckpt-13500.meta
└── summaries
    └── test_model_20
        └── events.out.tfevents.1521839337.db12gpu1.arc3.leeds.ac.uk
```

To train a new model, instead drop the `--training=False`. It is necessary to
add `--if_output_exists=continue` to continue training an already existing model.
The full set of runtime flags can be found in [run.py](CNN/run.py)

Flags of interest are as follows:

 - `training` - If the model should be trained or not.
 - `visualize` - If the PyGame GUI should be shown. Useful for local running during testing.

 - `n_envs` - Number of games to run in parallel. **Important** on a high powered machine to get the best performance.

 - `save_replays_every` - How often a game replay should be saved, such that the training progress can be seen.
 - `save_permanently_every` - Models are saved every few episodes, but are done in a rolling fashion. This flag is used to create models that will not be overwritten at any point
 - `curriculum_num` - What is the current curriculum number? Should be set for curriculum learning such that multiple models are loaded.
 - `previous_model` - Path to the previous model file, for curriculum learning.
 - `number_episodes` / `number_steps` - The maximum episodes or steps to take before stopping. If either of these are met, will stop.
