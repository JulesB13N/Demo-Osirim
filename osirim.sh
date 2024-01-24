#!/bin/bash
#SBATCH --job-name=Test_OSI   # Nom du Job  
#SBATCH --cpus-per-task=1    
#SBATCH --output=data.txt
#SBATCH --mail-type=END             # Notification par email de la
#SBATCH --mail-user=jules.bienassis@univ-tlse3.fr     # fin de l'exécution du job.

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch2-NGC-23-05-py3.sif $HOME/Demo-Osirim/venv/bin/python3 test.py