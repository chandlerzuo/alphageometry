#!/usr/bin/bash

# run this script with the following command:
# torchrun --no-python --nnodes 1 --nproc-per-node 2 ./run.sh

# Having run.sh file like this is preferred to running hello.py directly,
# because it allows us to add setup / teardown commands before the main python
# code is executed.

# gets the directory containing this script, cd, then runs hello.py
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR &&
python3 random_generation/generate_random_proofs.py
