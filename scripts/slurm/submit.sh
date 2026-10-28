#!/usr/bin/env bash
# Submit the pipeline for a group of regions to SLURM as four dependent jobs:
#
#   secnet     job array, one task per region           (independent of each other)
#   partition  one job, after every secnet task succeeded
#   primnet    job array, substations split over PRIMNET_TASKS tasks
#   finalnet   one job, after every primnet task succeeded
#
# Usage (from anywhere; jobs run in the repository root):
#   scripts/slurm/submit.sh [GROUP_CONFIG] [CONFIG] [-- EXTRA_MAIN_ARGS...]
#     GROUP_CONFIG  regions to run          (default: configs/regions/virginia.yaml)
#     CONFIG        algorithm configuration (default: configs/config_heuristic.yaml)
#     EXTRA_MAIN_ARGS are passed to every main.py call, e.g. -- --from-scratch
#
# Every setting below can also be given as an environment variable, e.g.
#   ACCOUNT=my_allocation PARTITION=short scripts/slurm/submit.sh configs/regions/northern_virginia.yaml
set -euo pipefail

# ---- Cluster settings -------------------------------------------------------
# Empty settings are not passed to sbatch.
ACCOUNT="${ACCOUNT:-}"        # allocation/account charged for the jobs (sbatch --account)
PARTITION="${PARTITION:-}"    # partition/queue (sbatch --partition)
QOS="${QOS:-}"                # quality of service (sbatch --qos)
SBATCH_EXTRA="${SBATCH_EXTRA:-}"  # any other sbatch options for every job, e.g. "--constraint=x"
RUN="${RUN:-uv run}"          # command that runs Python in the project environment

# ---- Resources per stage ----------------------------------------------------
SECNET_TIME="${SECNET_TIME:-12:00:00}"
SECNET_MEM="${SECNET_MEM:-16G}"
SECNET_CPUS="${SECNET_CPUS:-1}"
SECNET_MAX_RUNNING="${SECNET_MAX_RUNNING:-50}"     # regions running at the same time

PARTITION_TIME="${PARTITION_TIME:-24:00:00}"
PARTITION_MEM="${PARTITION_MEM:-128G}"

PRIMNET_TASKS="${PRIMNET_TASKS:-100}"              # array tasks the substations are split over
PRIMNET_TIME="${PRIMNET_TIME:-12:00:00}"
PRIMNET_MEM="${PRIMNET_MEM:-16G}"
PRIMNET_CPUS="${PRIMNET_CPUS:-1}"                  # raise for MILP solver threads
PRIMNET_MAX_RUNNING="${PRIMNET_MAX_RUNNING:-100}"

FINALNET_TIME="${FINALNET_TIME:-12:00:00}"
FINALNET_MEM="${FINALNET_MEM:-128G}"

# ---- Arguments --------------------------------------------------------------
GROUP="configs/regions/virginia.yaml"
CONFIG="configs/config_heuristic.yaml"
[[ $# -gt 0 && $1 != "--" ]] && { GROUP=$1; shift; }
[[ $# -gt 0 && $1 != "--" ]] && { CONFIG=$1; shift; }
[[ $# -gt 0 && $1 == "--" ]] && shift
EXTRA_ARGS="$*"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p logs/slurm
# main.py command line of a stage (the stage comes first: -r/-s take several values)
main() { local stage=$1; shift; echo "$RUN main.py $stage -c $CONFIG -g $GROUP $* $EXTRA_ARGS"; }

common=(--parsable --chdir="$REPO")
[[ -n $ACCOUNT ]] && common+=(--account="$ACCOUNT")
[[ -n $PARTITION ]] && common+=(--partition="$PARTITION")
[[ -n $QOS ]] && common+=(--qos="$QOS")
# shellcheck disable=SC2206
[[ -n $SBATCH_EXTRA ]] && common+=($SBATCH_EXTRA)

N_REGIONS=$($(main secnet --list) | wc -l | tr -d ' ')
if [[ $N_REGIONS -eq 0 ]]; then
    echo "No regions in $GROUP" >&2
    exit 1
fi
GROUP_NAME=$(basename "$GROUP" .yaml)

secnet=$(sbatch "${common[@]}" \
    --job-name="${GROUP_NAME}-secnet" --array="0-$((N_REGIONS - 1))%${SECNET_MAX_RUNNING}" \
    --time="$SECNET_TIME" --mem="$SECNET_MEM" --cpus-per-task="$SECNET_CPUS" \
    --output="logs/slurm/%x-%A_%a.out" \
    --wrap="$(main secnet --task-index \$SLURM_ARRAY_TASK_ID --task-count "$N_REGIONS")")

partition=$(sbatch "${common[@]}" --dependency="afterok:$secnet" \
    --job-name="${GROUP_NAME}-partition" \
    --time="$PARTITION_TIME" --mem="$PARTITION_MEM" \
    --output="logs/slurm/%x-%j.out" \
    --wrap="$(main partition)")

primnet=$(sbatch "${common[@]}" --dependency="afterok:$partition" \
    --job-name="${GROUP_NAME}-primnet" --array="0-$((PRIMNET_TASKS - 1))%${PRIMNET_MAX_RUNNING}" \
    --time="$PRIMNET_TIME" --mem="$PRIMNET_MEM" --cpus-per-task="$PRIMNET_CPUS" \
    --output="logs/slurm/%x-%A_%a.out" \
    --wrap="$(main primnet --task-index \$SLURM_ARRAY_TASK_ID --task-count "$PRIMNET_TASKS")")

finalnet=$(sbatch "${common[@]}" --dependency="afterok:$primnet" \
    --job-name="${GROUP_NAME}-finalnet" \
    --time="$FINALNET_TIME" --mem="$FINALNET_MEM" \
    --output="logs/slurm/%x-%j.out" \
    --wrap="$(main finalnet)")

echo "Submitted $GROUP ($N_REGIONS regions):"
echo "  secnet    $secnet"
echo "  partition $partition"
echo "  primnet   $primnet"
echo "  finalnet  $finalnet"
