
#$ -l h_rt=00:02:00
#$ -l coproc_k80=1
#$ -l ports=1

# Load modules
module add singularity/2.4
module add cuda/9.0.176

# Run program via Singularity
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H%M")
DATE_TIME=${DATE}_${TIME}

HOME_DIR="/home/ufaserv1_h/sc13rjc"

LOG_FILE="${HOME_DIR}/logs/pysc2_${DATE}_${TIME}.log"
STARCRAFT_IMG="/nobackup/sc13rjc/starcraft.simg"

SCRIPT_TO_RUN="${HOME_DIR}/git/meng_project/CNN/run.py"
SCRIPT_ARGS="--map_name MoveToBeacon --model_name test_model --n_envs 15"

echo "Starting script..."
singularity exec --nv ${STARCRAFT_IMG} python ${SCRIPT_TO_RUN} ${SCRIPT_ARGS} >> ${LOG_FILE} 2>&1
echo "Finished script."
