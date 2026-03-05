#!/bin/bash

# must be run as sudo
# USAGE: start_slurm.sh <NODE_TYPE> [<CONTOLLER_ADDRESSES>]
# - Where NODE_TYPE is one of follow values: controller, compute, login

set -ex

LOG_FILE="/var/log/provision/provisioning.log"
CONTROLLER_IP_VALUES=($2)

main() {
  echo "[INFO] START: Starting Slurm daemons"

  # The scripts are downloaded from the customer S3 bucket by HyperPod into
  # /tmp/<bucket-name>/, which is the working directory this script is
  # launched from.  Derive the path dynamically so it works regardless of
  # the bucket name.
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

  # /tmp may be cleared on reboot, so copy prolog/epilog to a persistent
  # location and make them executable there.
  # slurmctld (controller) validates executability at startup; slurmd
  # (compute) must also be able to exec them when a job runs.
  SLURM_SCRIPTS_DIR="/opt/slurm/etc/scripts"
  mkdir -p "$SLURM_SCRIPTS_DIR"
  for script in prolog.sh epilog.sh; do
      src="$SCRIPT_DIR/$script"
      dst="$SLURM_SCRIPTS_DIR/$script"
      if [ -f "$src" ]; then
          cp "$src" "$dst"
          chmod +x "$dst"
          echo "[INFO] Copied and made executable: $dst"
      else
          echo "[WARN] $src not found, skipping"
      fi
  done

  if [[ $1 == "controller" ]]; then
    echo "[INFO] This is a Controller node. Start slurm controller daemon..."

    # Inject Prolog/Epilog paths into slurm.conf before slurmctld reads it.
    # Point at the persistent copies under /opt/slurm/etc/scripts/ rather than /tmp.
    # Remove any pre-existing lines to avoid duplicates on re-runs, then append.
    sed -i '/^Prolog=/d;/^Epilog=/d' /opt/slurm/etc/slurm.conf
    printf '\n' >> /opt/slurm/etc/slurm.conf
    echo "Prolog=${SLURM_SCRIPTS_DIR}/prolog.sh"  >> /opt/slurm/etc/slurm.conf
    echo "Epilog=${SLURM_SCRIPTS_DIR}/epilog.sh"  >> /opt/slurm/etc/slurm.conf
    echo "[INFO] Added Prolog and Epilog to /opt/slurm/etc/slurm.conf"

    systemctl enable --now slurmctld

    mv /etc/systemd/system/slurmd{,_DO_NOT_START_ON_CONTROLLER}.service \
        || { echo "Failed to mask slurmd, perhaps the AMI already masked it?" ; }
  elif [[ $1 == "compute" ]] || [[ $1 == "login" ]]; then
    echo "[INFO] Running on $1 node. Start slurm daemon..."

    # Login nodes must still restart slurmd to fetch slurm.conf to /var/spool/slurmd/, however
    # slurmd won't run because slurm.conf does not contain login nodes.
    SLURMD_OPTIONS="--conf-server $CONTROLLER_IP_VALUES" envsubst < /etc/systemd/system/slurmd.service > slurmd.service
    mv slurmd.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable --now slurmd

    mv /etc/systemd/system/slurmctld{,_DO_NOT_START_ON_CONTROLLER}.service \
        || { echo "Failed to mask slurmctld, perhaps the AMI already masked it?" ; }
  fi

  echo "[INFO] Start Slurm Script completed"
}

main "$@"