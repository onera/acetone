#!/bin/bash

# Choice of a system
system="acasxu"
#system="acasxu_2"
#system="vcas_2"

# Path to the result files
path_traces="../simulation_traces_test/"

mkdir ${path_traces}

# Path to the list of the initial states to be simulated
path_initial_states="../init_states/init_states_"${system}".csv"

# Mode to compute the inference of the networks (schedmcore or ACETONE)
mode="schedmcore"

python3 main.py ${system} ${path_initial_states} ${path_traces} ${mode}







