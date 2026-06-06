#!/bin/bash
#SBATCH --job-name=q50-dqi-binary-gf5
#SBATCH --account=project_465003017
#SBATCH --reservation=JQH2026
#SBATCH --partition=debug
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=q50-binary-gf5-%j.out
module load Local-quantum
module load fiqci-vtt-qiskit-JQH
cd /scratch/project_465003017/rantajon/q50_immunization
# stage in this dir alongside the runner: q50_immunization_binary_gf5.qasm2 AND iqm_runner.py
python q50_run_binary_gf5.py
