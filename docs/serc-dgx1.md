# IISc SERC DGX-1 (`nvidia-dgx`) — reference notes

> Last updated: 2025-12-19  
> Source: IISc/SERC DGX-1 usage notes copied into this repo. Policies/queues can change; update this doc if you find drift.

See also:
- [`README.md`](../README.md) (hands-on workflow)
- [`INSTRUCTIONS.md`](../INSTRUCTIONS.md) (end-to-end walkthrough)

## Critical policies (don’t skip)

- **All compute must run via SLURM.** The official guidance states: “Running jobs without SLURM will lead to blocking of the computational account”. Some published job-submission notes also warn that running outside SLURM may be escalated and can result in account blocking.
- **`/localscratch` is temporary.** Data older than **~14 days (2 weeks)** may be deleted; SERC does not maintain backups.
- **Docker images are also temporary.** Docker images older than **~14 days (2 weeks)** may be deleted without notice; keep Dockerfiles and/or backups.

## What DGX-1 is (high-level)

DGX-1 is an 8×GPU deep learning system designed for high-throughput training, with high-bandwidth GPU↔GPU interconnect.

## Hardware overview (as published)

| Component | DGX-1 |
|---|---|
| GPUs | 8 × Tesla V100 |
| GPU memory | 256 GB total system |
| CPU | Dual 20-core Intel Xeon E5-2698 v4 @ 2.2 GHz |
| CUDA cores | 40,960 |
| Tensor cores | 5,120 (V100-based systems) |
| System memory | 512 GB DDR4 RDIMM (2.133 GHz) |
| Storage | 4 × 1.92 TB SSD (RAID-0) |
| Network | Dual 10 GbE |

### Vendors (as published)

- OEM: NVIDIA Corporation
- Authorized seller: LOCUZ Enterprise Solutions Ltd

### NVLink (as published)

NVLink enables GPU↔GPU data exchange at high bandwidth. The published DGX-1 notes describe an aggregate bi-directional bandwidth of up to **300 GB/s per GPU**, and a “hybrid cube-mesh” GPU network topology.

### Performance (as published)

The published DGX-1 notes include “1 PFLOPS (mixed precision)” and the following table:

| Tesla V100 (NVLink) performance | Single V100 GPU | Total (8 × V100) |
|---|---:|---:|
| Single precision | Up to 7.8 TFLOPS | Up to 62.4 TFLOPS |
| Double precision | Up to 15.7 TFLOPS | Up to 125.6 TFLOPS |
| Deep learning (mixed precision) | Up to 125 TFLOPS | Up to 1 PFLOPS |

## Software overview (as published)

- OS: Ubuntu 16.04 (Linux x86_64)
- Workload manager / job submission: SLURM
- Environments: Docker is enabled (recommended for per-user dependencies)

## Accessing DGX-1

### Login node

DGX-1 has one login node: `nvidia-dgx`. Login is via SSH **from inside the IISc network**:

```bash
ssh <computational_userid>@nvidia-dgx.serc.iisc.ac.in
```

### Getting an account (as published)

1. Apply for basic HPC access by filling the **computational account** form and emailing it to `nisadmin.serc@iisc.ac.in`.
2. The HPC application form must be signed by your advisor / research supervisor.
3. After the computational account is created, fill the **NVIDIA DGX access** form to access DGX.

## Storage layout and retention

### Home

Home directories are typically limited to ~**1.5 GB**.

### `localscratch` (where you should work)

Your working directory is:

```bash
cd /localscratch/<computational_userid>
```

Retention policy (as published):

- `/localscratch` data older than **~14 days (2 weeks)** may be deleted.
- SERC does not maintain backups and is not responsible for data loss.

Practical implication:

- Treat `/localscratch` like a cache: stage data in at job start (if feasible), stage results out frequently.
- Keep experiment-critical data and long-term artifacts in persistent storage you control.

## SLURM quick reference

Common commands (as published):

- Submit: `sbatch <script.sbatch>`
- Queue status: `squeue` (or `squeue <job_id>`)
- Partitions/nodes: `sinfo`
- Cancel a job: `scancel <job_id>`
- The published notes mention “kill job” as `kill <JOB ID>`. In practice, prefer **`scancel <job_id>`** (the `kill` command targets process IDs, not SLURM job IDs).

## DGX-1 queue (partition) configuration (as published)

Six “regular” queues are listed:

| Queue (partition) | GPUs | Wall time |
|---|---:|---:|
| `q_1day-1G` | 1 | 24 hours |
| `q_2day-1G` | 1 | 48 hours |
| `q_1day-2G` | 2 | 24 hours |
| `q_2day-2G` | 2 | 48 hours |
| `q_1day-4G` | 4 | 24 hours |
| `q_2day-4G` | 4 | 48 hours |

## Job script pattern: SLURM → Docker

The typical pattern is:

1. SLURM allocates GPUs/CPU/RAM/time.
2. Your `sbatch` script runs `docker run` bound to the SLURM allocation.
3. You bind-mount your scratch directory into the container and run your code there.

This repo includes working examples under `slurm/*.sbatch`. For a minimal skeleton:

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=q_1day-1G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

USER_NAME="${USER}"
SCRATCH="/localscratch/${USER_NAME}"

docker run --rm \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --ipc=host --shm-size=20G \
  --user "$(id -u ${USER_NAME}):$(id -g ${USER_NAME})" \
  -v "${SCRATCH}:${SCRATCH}" \
  -w "${SCRATCH}/<your_project_dir>" \
  "<your_image>:<tag>" \
  bash -lc "python <your_script>.py"
```

## Docker usage notes

### Principle: match UID/GID

To avoid permission issues on bind-mounted directories, the published guidance recommends creating a user inside the container with the **same UID/GID** as your computational account.

This repo’s `Dockerfile.modern`/`Dockerfile.compat` already follow this idea via build args (`UID`, `GID`, `USERNAME`).

### “Docker usage inside DGX-1” sample (as published)

The published DGX-1 notes include an example approach:

1. Start from a CUDA base image (example: `nvidia/cuda:11.0.3-devel-ubuntu20.04`).
2. Get your UID/GID on DGX:

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
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

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

### Debugging: exec into a running container (as published)

If you launched a long-running container and want to inspect it:

```bash
docker ps
docker exec -u <dockerusername> -it <CONTAINER_ID> bash
```

The published notes also mention common conda setup steps:

```bash
conda --version
conda init
source ~/.bashrc
conda deactivate
```

## Backing up Docker images

Method 1 (as published): save/load tarballs:

```bash
docker image list
docker save -o /path/to/backup/your-image.tar your-image:tag
docker load -i /path/to/backup/your-image.tar
```

Because `/localscratch` is also purged, copy image tarballs to persistent storage.

Method 2 (as published): push images to a registry (Docker Hub / private registry).

## Support and contact (as published)

- Preferred: raise a ticket on the SERC HelpDesk (better tracking).
- Email (less preferred): `helpdesk.serc@auto.iisc.ac.in` with a suitable subject line.
- For DGX-1 issues: contact the system administrator, room `#103`, SERC (CPU room, ground floor).

## Location (as published)

- CPU room — ground floor, SERC, IISc.

## Announcements (historical)

The published notes included: “New SERC Power Shutdown – December 24-27, 2025”.
