#$ -l h_rt=01:00:00
#$ -l coproc_k80=1
#$ -l ports=5

# Load modules
module add singularity/2.4
module add cuda/9.0.176

# Run program via Singularity
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H%M")
DATE_TIME=${DATE}_${TIME}

NO_BACKUP_DIR="/nobackup/sc13rjc"
LOG_FILE="${NO_BACKUP_DIR}/logs/pysc2_${DATE}_${TIME}.log"

PROJECT_DIR="${NO_BACKUP_DIR}/git/meng_project/CNN"
STARCRAFT_IMG="${NO_BACKUP_DIR}/starcraft.simg"

MAP_NAME="MoveToBeacon"
AGENT_NAME="test_model_1"

SCRIPT_TO_RUN="${PROJECT_DIR}/run.py"
SCRIPT_ARGS="--map_name ${MAP_NAME} --model_name ${AGENT_NAME} --n_envs 64 --if_output_exists continue"

echo "Starting script..."
echo "Swapping to ${PROJECT_DIR}"
echo "${MAP_NAME}" >> ${LOG_FILE}
echo "${AGENT_NAME}" >> ${LOG_FILE}
cd ${PROJECT_DIR}
singularity exec --nv -H ${NO_BACKUP_DIR} ${STARCRAFT_IMG} python ${SCRIPT_TO_RUN} ${SCRIPT_ARGS} >> ${LOG_FILE} 2>&1
echo "Finished script."

