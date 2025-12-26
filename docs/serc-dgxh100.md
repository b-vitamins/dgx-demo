# IISc SERC DGX-H100 (`dgxh100`) — reference notes

> Last updated: 2025-12-19  
> Source: IISc/SERC DGX-H100 usage notes copied into this repo. Policies/queues can change; update this doc if you find drift.

See also:
- [`README.md`](../README.md) (hands-on workflow)
- [`INSTRUCTIONS.md`](../INSTRUCTIONS.md) (end-to-end walkthrough)
- [`docs/dgxh100-adaptation.md`](dgxh100-adaptation.md) (how to adapt this repo to DGX-H100)

## Critical policies (don’t skip)

- **All compute must run via SLURM.** The official guidance states: “Running jobs without SLURM will lead to blocking of the computational account”. Some published job-submission notes also warn that running outside SLURM may be escalated and can result in account blocking.
- **`/raid` is temporary.** Data older than **~14 days (2 weeks)** may be deleted; SERC does not maintain backups.
- **Docker images are also temporary.** Docker images older than **~14 days (2 weeks)** may be deleted without notice; keep Dockerfiles and/or backups.

## What DGX-H100 is (high-level)

DGX H100 is an 8×H100 system designed for large-scale training/inference, with NVLink/NVSwitch providing very high GPU↔GPU bandwidth.

## Hardware overview (as published)

| Component | DGX H100 |
|---|---|
| GPUs | 8 × NVIDIA H100 (640 GB total GPU memory) |
| CPU | 2 × Intel Xeon Platinum 8480C (56 cores each, 112 cores total) |
| System memory | 2 TB |
| NVLink | 18 NVLink connections per GPU; up to 900 GB/s bidirectional GPU↔GPU bandwidth (published) |
| NVSwitch | 4 × NVIDIA NVSwitches; 7.2 TB/s bidirectional GPU↔GPU bandwidth (published) |
| Storage (OS) | 2 × 1.92 TB NVMe M.2 (RAID-1) |
| Storage (data cache) | 8 × 3.84 TB NVMe U.2 SED (RAID-0) |
| Network (cluster) | ConnectX-7 adapters (InfiniBand up to 400 Gbps / Ethernet up to 400 GbE, published) |
| Network (storage + in-band mgmt) | ConnectX-7 dual-port Ethernet (published) |
| BMC | 1 GbE RJ45 (Redfish/IPMI/SNMP/KVM/Web UI, published) |
| Power | 6 × 3.3 kW (published) |

### Vendors (as published)

- OEM: NVIDIA Corporation
- Vendor: Frontier

Mechanical specs (as published):

| Feature | Value |
|---|---|
| Form factor | 8U rackmount |
| Height | 14” (356 mm) |
| Width | 19” (482.3 mm) max |
| Depth | 35.3” (897.1 mm) max |
| System weight | 287.6 lbs (130.45 kg) max |

## Software overview (as published)

- OS: Ubuntu 22.04.2 LTS (Linux x86_64)
- Workload manager / job submission: SLURM

## Accessing DGX-H100

### Login node

DGX-H100 has one login node: `dgxh100`. Login is via SSH **from inside the IISc network**:

```bash
ssh <computational_userid>@dgxh100.serc.iisc.ac.in
```

### Getting an account (as published)

1. Apply for basic HPC access by filling the **computational account** form and emailing it to `nisadmin.serc@iisc.ac.in`.
2. The HPC application form must be signed by your advisor / research supervisor.
3. After the computational account is created, fill the **NVIDIA DGXH100 access** form to access DGX-H100.

## Storage layout and retention

### Home

Home directories are typically limited to ~**1.5 GB**.

### `raid` (where you should work)

Your working directory is:

```bash
cd /raid/<computational_userid>
```

Retention policy (as published):

- `/raid` data older than **~14 days (2 weeks)** may be deleted.
- SERC does not maintain backups and is not responsible for data loss.

## SLURM quick reference

Common commands (as published):

- Submit: `sbatch <script.sbatch>`
- Queue status: `squeue`
- Partitions/nodes: `sinfo`
- Cancel a job: `scancel <job_id>`
- The published DGX-1 notes mention “kill job” as `kill <JOB ID>`. In practice, prefer **`scancel <job_id>`** (the `kill` command targets process IDs, not SLURM job IDs).

## DGX-H100 queue (partition) configuration (as published)

Five “regular” queues are listed:

| Queue (partition) | GPUs | Wall time | Notes |
|---|---:|---:|---|
| `q_12hour-1G` | 1 | 12 hours | Prototyping |
| `q_1day-1G` | 1 | 24 hours | Production |
| `q_2day-1G` | 1 | 48 hours | Production |
| `q_1day-2G` | 2 | 24 hours | Production |
| `q_2day-2G` | 2 | 48 hours | Production |

## Job script pattern: SLURM → Docker

The typical pattern is:

1. SLURM allocates GPUs/CPU/RAM/time.
2. Your `sbatch` script runs `docker run` bound to the SLURM allocation.
3. You bind-mount your `/raid/<user>` directory into the container and run your code there.

Minimal skeleton:

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=q_1day-1G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

USER_NAME="${USER}"
RAID="/raid/${USER_NAME}"

docker run --rm \
  --gpus '"device='"${CUDA_VISIBLE_DEVICES}"'"' \
  --ipc=host --shm-size=20G \
  --user "$(id -u ${USER_NAME}):$(id -g ${USER_NAME})" \
  -v "${RAID}:${RAID}" \
  -w "${RAID}/<your_project_dir>" \
  "<your_image>:<tag>" \
  bash -lc "python <your_script>.py"
```

## Docker usage notes

### Principle: match UID/GID

To avoid permission issues on bind-mounted directories, the published guidance recommends creating a user inside the container with the **same UID/GID** as your computational account.

### “Docker usage inside DGXH100” sample (as published)

The published DGX-H100 notes include an example approach:

1. Start from a CUDA base image (example: `nvidia/cuda:12.2.0-devel-ubuntu20.04`).
2. Get your UID/GID on DGXH100:

   ```bash
   id <computational_userid>
   ```

3. Create a user in the container that matches your DGX user (UID/GID), and install the packages you need (Miniconda is one common choice).
4. Build your image (note the trailing `.`):

   ```bash
   docker build -t <preferred_docker_image_name> .
   docker image list
   ```

5. Run your image via SLURM.

Example Dockerfile (verbatim structure, with placeholders):

```Dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV dockerusername=secdsan
ENV dockeruserpassword=password
ENV dockerusergroupid=1040
ENV dockeruserid=18308
ENV dockerusergroupname=serc3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget bzip2 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/miniconda/bin:$PATH

RUN apt-get update && apt-get install -y sudo

RUN groupadd $dockerusergroupname -g $dockerusergroupid
RUN useradd $dockerusername -u $dockeruserid -g $dockerusergroupid -d /home/$dockerusername
RUN mkdir -p /home/$dockerusername && chown -R $dockerusername:$dockerusergroupid /home/$dockerusername
RUN echo "$dockerusername:$dockeruserpassword" | chpasswd
RUN usermod -aG sudo $dockerusername

USER $dockerusername
```

### Debugging inside a running container (as published)

The published notes mention:

```bash
docker ps
docker exec -u <dockerusername> -it <CONTAINER_ID> bash
```

Then install packages with `sudo` (password is the one configured for the container user), and run code.

## Backing up Docker images

Method 1 (as published): save/load tarballs:

```bash
docker image list
docker save -o /path/to/backup/your-image.tar your-image:tag
docker load -i /path/to/backup/your-image.tar
```

Because `/raid` is also purged, copy image tarballs to persistent storage.

Method 2 (as published): push images to a registry (Docker Hub / private registry).

## Support and contact (as published)

- Preferred: raise a ticket on the SERC HelpDesk (better tracking).
- Email (less preferred): `helpdesk.serc@auto.iisc.ac.in` with a suitable subject line.

## Location (as published)

- CPU room — ground floor, SERC, IISc.
