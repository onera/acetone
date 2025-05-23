'''
This file is part of a program that allows simulating the ACAS Xu and VCAS neural network controlled systems
Copyright (c) 2023 Arthur Claviere ONERA/ISAE-SUPAERO

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
'''


def simulate_one_step(system, nnets_dict, plant_state, prev_labels, mode):
    """ Perform a one-step simulation of the system of interest:
    - calculate the commands to be applied to the plant, depending on the previous labels of the agents/controllers composing the NNCS
    - simulate the plant dynamics
    - for each agent/controller:
        (i)   pre-process the sampled state of the plant
        (ii)  select the neural network to be executed, depending on the previous label
        (iii) normalize the inputs of the network
        (iv)  execute the neural network
        (v)   post-process the scores produced by the network so as to determine the previous label for next step


    Parameters
    ----------
    system : AbstractProductSystem object
        The system of interest
    nnets_dict : dictionary
        The neural networks composing the controller, nnets_dict[nnet_name] is a FFNN object
    plant_state : 1D numpy array
        The state of the plant
    prev_labels : list
        The previous labels of the different agents/controllers composing the NNCS
    mode : str
        The mode to compute the inference of the networks (schedmcore or ACETONE)

    Return
    ------
    tuple
        A tuple containing:
            - the safety indicator
            - the state of the plant at the end of the simulation
            - the non-normalized input of the network
            - the name of the network that was executed
            - the normalized input of the network
            - the output scores from the neural network (1D numpy array)
            - the previous label for next step
    """
    # Calculate the commands to be applied to the plant
    command = system.prev_labels_to_commands(prev_labels)

    # Evaluate the plant dynamics
    (plant_state_new) = system.simulate_dynamics_euler(plant_state, command)
    is_safe = system.is_safe_state(plant_state_new)

    # Retrieve the number of agents/controllers composing the NNCS
    n_agents = system.get_n_agents()

    # For each agent, calculate the new label
    prev_labels_new = []
    nnet_names = []
    nnet_inputs = []
    nnet_inputs_norm = []
    scores = []
    for id_agent in range(n_agents):
        # (i) pre-process the sampled state of the plant
        in_vector = system.pre_process(id_agent, plant_state)
        nnet_inputs.append(in_vector)

        # (ii) select the neural network to be executed, depending on the previous label
        nnet_name = system.select_network(id_agent, prev_labels[id_agent])
        nnet_names.append(nnet_name)

        # (iii) normalize the input set
        in_norm = system.normalize_input(nnet_name, in_vector)
        nnet_inputs_norm.append(in_norm)

        # (iv) evaluate the neural network
        score = nnets_dict[nnet_name].compute_output(in_norm, mode)
        scores.append(score)

        # (v) post-process the scores produced by the network so as to determine the new label
        label = system.post_process(id_agent, score)
        prev_labels_new.append(label)

    return (is_safe, plant_state_new, nnet_inputs, nnet_names, nnet_inputs_norm, scores, prev_labels_new)


def simulate(system, nnets_dict, plant_state_init, prev_labels_init, n_steps, mode):
    """ Perform a simulation of the system of interest, starting from a given scalar intial state represented by the 2-tuple (plant_state_init, prev_labels_init).

    Parameters
    ----------
    system : AbstractProductSystem object
        The system of interest
    nnets_dict : dictionary
        The neural networks composing the controller, nnets_dict[nnet_name] is a FFNN object object
    plant_state_init : 1D numpy array
        The initial state of the plant
    prev_labels_init : int
        The initial previous labels of the agents/controllers composing the NNCS
    n_steps : int
        The upper bound on the number of simulation steps to be performed
    mode : str
        The mode to compute the inference of the networks (schedmcore or ACETONE)

    Return
    ------
    tuple
        A tuple contaning:
            - a boolean indicating whether the trace is safe or not
            - the simulation trace [[x_k, prev_labels_k, nnet_inputs_k, nnet_names_k, nnet_inputs_norm_k, scores_k, ],...]
    """
    is_safe = True
    n_simulation_steps = 0
    plant_state = plant_state_init
    prev_labels = prev_labels_init
    trace = []
    while is_safe and n_simulation_steps < n_steps:
        n_simulation_steps += 1
        # one-step simulation
        (is_safe, plant_state_new, nnet_inputs, nnet_names, nnet_inputs_norm, scores,
         prev_labels_new) = simulate_one_step(system, nnets_dict, plant_state, prev_labels,mode)
        trace.append((plant_state, prev_labels, nnet_inputs, nnet_names, nnet_inputs_norm, scores))
        plant_state = plant_state_new
        prev_labels = prev_labels_new
    return (is_safe, trace)
