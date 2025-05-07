'''
This file is part of a program that allows simulating the ACAS Xu and VCAS neural network controlled systems
Copyright (c) 2023 Arthur Claviere ONERA/ISAE-SUPAERO

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
'''

# add path to local packages and code
import sys

sys.path.append('mlmodels/')
sys.path.append('systems/')

# imports
import configparser
import argparse
import os
import shutil
import math
import pandas as pd
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# local imports
from .simulation import simulate
from .systems.system import *
from .systems.system_acasxu import *
from .systems.system_acasxu_2 import *
from .systems.system_vcas_2 import *


def array_to_str(array):
    res = '['
    for k in range(array.size):
        res += str(array[k])
        if k < array.size - 1:
            res += ' '
    res += ']'
    return res


def str_to_array(list_str):
    temp_str = list_str.split('[')[1]
    temp_str = temp_str.split(']')[0]
    values = temp_str.split(',')
    return np.array([float(v) for v in values])


def str_to_int_list(list_str):
    temp_str = list_str.split('[')[1]
    temp_str = temp_str.split(']')[0]
    values = temp_str.split(',')
    return [int(v) for v in values]

def get_coordinates(trace, delta_t):
    x_own = [0]
    y_own = [0]
    x_intruder = [trace[0][0][0]]
    y_intruder = [trace[0][0][1]]

    for k in range(len(trace)):
        plan_state = trace[k][0]
        psi_own = plan_state[2]
        v_own = plan_state[4]
        psi_intruder = plan_state[3]
        v_intruder = plan_state[5]

        x_own.append(x_own[k] - delta_t*v_own*np.sin(psi_own))
        y_own.append(y_own[k] + delta_t*v_own*np.cos(psi_own))
        x_intruder.append(x_intruder[k] - delta_t*v_intruder*np.sin(psi_intruder))
        y_intruder.append(y_intruder[k] + delta_t*v_intruder*np.cos(psi_intruder))

    return x_own, y_own, x_intruder, y_intruder


def run_simulation(system_name, path_initial_states, directory_results, mode):
    # Instantiate the configuration parsers and read the systems config files
    config_systems = configparser.ConfigParser()
    config_systems.read('./schedmcore_ACAS_CaseStudy/src/config_systems.ini')

    # System
    if system_name == 'acasxu':
        # Open the corresponding section in config_systems
        config_system = config_systems['ACASXU']
        # Define the corresponding system factory
        system_factory = ConcreteFactoryACASXU()
    if system_name == 'acasxu_2':
        # Open the corresponding section in config_systems
        config_system = config_systems['ACASXU_2']
        # Define the corresponding system factory
        system_factory = ConcreteFactoryACASXU_2()
    if system_name == 'vcas_2':
        # Open the corresponding section in config_systems
        config_system = config_systems['VCAS_2']
        # Define the corresponding system factory
        system_factory = ConcreteFactoryVCAS2()

    # (I) Create a DataFrame based on the .csv file containing the initial states to be simulated
    df_initial_states = pd.read_csv(path_initial_states)

    # (II) Create the system object
    system = system_factory.createSystem(config_system)
    n_agents = system.get_n_agents()

    # (III) Read the neural networks and create a dictionary with the neural networks
    nnets = system.read_networks()
    nnets_dict = {}
    for (nnet_name, nnet, path_nnet) in nnets:
        nnets_dict[nnet_name] = nnet

    # (IV) Simulate each initial state in df_initial_states
    n_initial_states = len(df_initial_states.index)
    for n in range(n_initial_states):

        # Retrieve useful information for the simulation
        t_end = float(df_initial_states.iat[n, 0])
        x_p_0 = str_to_array(df_initial_states.iat[n, 1])
        x_c_0 = str_to_int_list(df_initial_states.iat[n, 2])

        # Retrieve the number of steps to be performed
        execution_period = config_system.getfloat('Controllers_execution_period')
        n_steps = int(t_end / execution_period)

        # Perform simulation
        (is_safe, trace) = simulate(system, nnets_dict, x_p_0, x_c_0, n_steps,mode)
        # record the trace as a json file
        data = {}
        for k in range(len(trace) - 1):
            time_step_id = str(k)
            data[time_step_id] = {}
            for id_agent in range(n_agents):
                prev_adv = trace[k][1][id_agent]
                x_in = trace[k][2][id_agent]
                nnet_name = trace[k][3][id_agent]
                x_in_norm = trace[k][4][id_agent]
                y = trace[k][5][id_agent]
                adv = trace[k + 1][1][id_agent]
                data[time_step_id]['agent_{}'.format(id_agent)] = {}
                data[time_step_id]['agent_{}'.format(id_agent)]['1_name_nnet'] = nnet_name
                data[time_step_id]['agent_{}'.format(id_agent)]['2_in'] = array_to_str(x_in)
                data[time_step_id]['agent_{}'.format(id_agent)]['3_in_norm'] = array_to_str(x_in_norm)
                data[time_step_id]['agent_{}'.format(id_agent)]['4_out'] = array_to_str(y)
                data[time_step_id]['agent_{}'.format(id_agent)]['5_decision'] = str(adv)
        json_data = json.dumps(data, indent=4, separators=(',', ': '))
        filename_json = 'trace_' + system_name + '_init_state_{}.json'.format(n)
        path_trace_json = directory_results + filename_json
        with open(path_trace_json, 'w') as f_json:
            f_json.write(json_data)

        x_own, y_own, x_intruder, y_intruder = get_coordinates(trace, execution_period)
        plt.plot(x_intruder, y_intruder, label='Intruder')
        plt.plot(x_own, y_own, label='Own')
        plt.legend()
        plt.show()

    clean_network(nnets)


def main():
    # Instantiate the argument parser
    parser = argparse.ArgumentParser(
        description="Simple program for simulating a neural network controlled system (NNCS) e.g., the ACAS Xu."
    )
    # Argument parser
    # Define restricted choices
    available_systems = ["acasxu", "acasxu_2", "vcas_2"]
    # Define required positional arguments
    parser.add_argument(
        "system_name",
        type=str,
        choices=available_systems,
        metavar="system_name",
        help="The name of the NNCS (e.g., acasxu).",
    )
    parser.add_argument(
        "path_initial_states",
        type=str,
        help="The path to the .csv file containing the initial states for the simulations.",
    )
    parser.add_argument(
        "directory_results",
        type=str,
        help="The directory where the result files (the traces) are to be recorded.",
    )
    parser.add_argument(
        "mode",
        type=str,
        help="TThe mode to compute the inference of the networks (schedmcore or ACETONE).",
    )

    # Parse the arguments
    args = parser.parse_args()

    run_simulation(
        args.system_name, args.path_initial_states, args.directory_results, args.mode
    )

if __name__ == "__main__":
    main()
