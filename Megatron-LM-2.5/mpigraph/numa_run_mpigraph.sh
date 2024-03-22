#!/bin/bash

# Global settings
export MONITOR_GPU=1
export OMP_PROC_BIND=TRUE 
export OMP_PLACES=sockets

usage() {
  echo ""
  echo "$(basename $0) [OPTION]"
  echo "    --bin    <string>             name of binary with absolute path"
  echo "    --config <string>             name of config with preset options ('HPE-a100'),"
  echo "                                  or path to a shell file"
  echo "    --option <string>             options for lammps run"
  echo "Examples:"
  echo " mpirun ... $(basename $0) --config HPE-a100 --option ..."
}

info() {
  local msg=$*
  echo -e "INFO: ${msg}"
}

warning() {
  local msg=$*
  echo -e "WARNING: ${msg}"
}

error() {
  local msg=$*
  echo -e "ERROR: ${msg}"
  exit 1
}

set_config() {
      CPU_AFFINITY="8-15:24-31:40-47:56-63"
      GPU_AFFINITY="3:1:7:5"
      MEM_AFFINITY="1:3:5:7"
      NET_AFFINITY="mlx5_0:mlx5_1:mlx5_2:mlx5_5"
      CPU_CORES_PER_RANK=8
}

read_rank() {
  # Global rank
  if [ -n "${OMPI_COMM_WORLD_RANK}" ]; then
    RANK=${OMPI_COMM_WORLD_RANK}
  elif [ -n "${PMIX_RANK}" ]; then
    RANK=${PMIX_RANK}
  elif [ -n "${PMI_RANK}" ]; then
    RANK=${PMI_RANK}
  elif [ -n "${SLURM_PROCID}" ]; then
    RANK=${SLURM_PROCID}
  else
    warning "could not determine rank"
  fi

  # Node local rank
  if [ -n "${OMPI_COMM_WORLD_LOCAL_RANK}" ]; then
    LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
  elif [ -n "${SLURM_LOCALID}" ]; then
    LOCAL_RANK=${SLURM_LOCALID}
  else
    error "could not determine local rank"
  fi
}

# split the affinity string, e.g., '0:2:4:6' into an array,
# e.g., map[0]=0, map[1]=2, ...  The array index is the MPI rank.
read_cpu_affinity_map() {
    local affinity_string=$1
    readarray -t CPU_AFFINITY_MAP <<<"$(tr ':' '\n'<<<"$affinity_string")"
}

# split the affinity string, e.g., '0:2:4:6' into an array,
# e.g., map[0]=0, map[1]=2, ...  The array index is the MPI rank.
read_gpu_affinity_map() {
    local affinity_string=$1
    readarray -t GPU_AFFINITY_MAP <<<"$(tr ':' '\n'<<<"$affinity_string")"
}

# split the affinity string, e.g., '0:2:4:6' into an array,
# e.g., map[0]=0, map[1]=2, ...  The array index is the MPI rank.
read_mem_affinity_map() {
    local affinity_string=$1
    readarray -t MEM_AFFINITY_MAP <<<"$(tr ':' '\n'<<<"$affinity_string")"
}

# split the affinity string, e.g., '0:2:4:6' into an array,
# e.g., map[0]=0, map[1]=2, ...  The array index is the MPI rank.
read_net_affinity_map() {
    local affinity_string=$1
    readarray -t NET_AFFINITY_MAP <<<"$(tr ':' '\n'<<<"$affinity_string")"
}

RUN_BIN=$2
OPT="$3 $4 $5"
set_config

if [ -z "$CPU_AFFINITY" ] || [ -z "${GPU_AFFINITY}" ]; then
  error "necessary parameters not set, see $(basename $0) --help"
fi

# Figure out the right parameters for this particular rank
read_rank
read_cpu_affinity_map $CPU_AFFINITY
read_mem_affinity_map $MEM_AFFINITY
read_net_affinity_map $NET_AFFINITY

CPU=${CPU_AFFINITY_MAP[$LOCAL_RANK]}
MEM=${MEM_AFFINITY_MAP[$LOCAL_RANK]}
NET=${NET_AFFINITY_MAP[$LOCAL_RANK]}


if [ -n "${NET}" ]; then
  export UCX_NET_DEVICES="$NET:1"
  echo $UCX_NET_DEVICES
fi
if [ -n "${MEM}" ]; then
  MEMBIND="--membind=${MEM}"
fi

#echo "CPU $CPU"
#echo "MEMBIND $MEMBIND"
info "host=$(hostname) rank=${RANK} lrank=${LOCAL_RANK} cores=${CPU_CORES_PER_RANK} gpu=${GPU} cpu=${CPU} net=${UCX_NET_DEVICES}"
#echo "numactl --physcpubind=${CPU} ${MEMBIND} ${RUN_BIN} ${OPT}"
numactl --physcpubind=${CPU} ${MEMBIND} ${RUN_BIN} ${OPT} 
