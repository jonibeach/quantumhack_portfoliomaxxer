#!/bin/bash
#SBATCH --job-name=q50-bell
#SBATCH --account=project_465003017
#SBATCH --reservation=JQH2026
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=q50-%j.out
module load Local-quantum
module load fiqci-vtt-qiskit-JQH
python bell_state.py
