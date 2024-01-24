#!/bin/bash
#SBATCH --job-name=Test_OSI   # Nom du Job  
#SBATCH --cpus-per-task=1    
#SBATCH --ouput=data.txt
#SBATCH --mail-type=END             # Notification par email de la
#SBATCH --mail-user=jules.bienassis@univ-tlse3.fr     # fin de l'exécution du job.

python3 test.py