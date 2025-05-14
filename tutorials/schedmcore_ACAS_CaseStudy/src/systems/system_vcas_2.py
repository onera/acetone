'''
This file is part of a program that allows simulating the ACAS Xu and VCAS neural network controlled systems
Copyright (c) 2023 Arthur Claviere ONERA/ISAE-SUPAERO

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
'''

from abc import ABC, abstractmethod
from math import *
import numpy as np
import subprocess
import os
import shutil

# local imports
from .system import *
from ..mlmodels.ffnn import *
from ..utils import *

class ConcreteProductSystemVCAS2(AbstractProductSystem):
    """A class corresponding to the VCAS2 system and the associated simulation options, implementing the AbstractProductSystem class

    Attributes (Specific)
    ----------
    norm_parameters
        A dictionary recording the normalization parameters for each network composing the NNCS
    prefix_nnet_names
        The prefix of the nnet names
    available_commands
        The list of the commands which can be applied to the plant
    collision_radius
        The collision radius of the aircraft, in ft
    
    """   
    def __init__(self, config_system):
        """Constructor"""
        # call the super construstor
        super().__init__(config_system)
        # The dictionary where normalization parameters are to be stored
        self.norm_parameters = {}
        # The prefix of the nnet names
        self.prefix_nnet_names = 'nnet_vcas_'
        # The list of the available commands (to be applied to the plant)
        self.available_commands = [0.0,-7.33,7.33,-9.33,9.33,-9.7,9.7,-11.7,11.7] # [Clear-of-conflict, Weak left, Weak right, Strong left, Strong Right], measured in degree, counter clockwise
        # The collision radius
        self.collision_radius = 100.0
        
    def get_prefix_nnet_names(self) -> str:
        """Get the prefix for the nnet_names
        
        Parameters
        ----------
        None

        Return
        ------
        string
            The prefix of the nnet names
        
        """
        return self.prefix_nnet_names
        
    def read_networks(self) -> list:
        """Read the neural networks of all the agents/controllers composing the NNCS

        Parameters
        ----------
        None

        Return
        ------
        list
            A list of tuples (nnet_name, nnet, path_nnet), where nnet_name is the name of the network nnet is a FFNN object representing the network and path_nnet is the path to the original .nnet file

        """
        nnets =[]
        for i in range(1,10):
            # retrieve the path to the i^th network
            nnet_filename = 'VertCAS_pra0{}_v4_45HU_200.nnet'.format(i)
            path_nnet = self.path_nnets + nnet_filename
            # parse the network
            (nnet, norm_params) = parse_nnet_format(path_nnet, 'relu')
            # name the network
            nnet_name = self.prefix_nnet_names + str(i-1)
            # append the network to the nnets list
            nnets.append((nnet_name, nnet, path_nnet)) 
            # update the norm_parameters dictionary
            self.norm_parameters[nnet_name] = norm_params
        return nnets
        
    def is_safe_state(self, plant_state) -> bool:
        """Determine whether the state of the plant is safe
        
        Parameters
        ----------
        plant_state : 1D numpy array
            The state of the plant

        Return
        ------
        bool
            The boolean indicating whether the state of the plant is safe
        """
        return (abs(plant_state[3]) >= 0.001 or abs(plant_state[0]) >= self.collision_radius)
        
    def prev_labels_to_commands(self, prev_labels) -> np.array:
        """Convert the previous labels of the agents/controllers into a set of commands for the plant

        Parameters
        ----------
        prev_labels : list
            The previous labels of the agents/controllers composing the NNCS

        Return
        ------
        1D numpy array
            The corresponding commands for the plant
        """
        # convert turn rate into radian
        vert_acc = []
        for prev_label in prev_labels:
            vert_acc.append(self.available_commands[prev_label])
        return np.array(vert_acc)
        
    def simulate_dynamics_euler(self, plant_state, command) -> np.array:
        """Simulate the dynamics using Euler's method, for the execution period of the controller
        
        Parameters
        ----------
        plant_state : 1D numpy array
            The state of the plant (a point and not a hyperrectangle)
        command : 1D numpy array
            The command to be applied to the plant

        Return
        ------
        1D numpy array
            The state of the plant after the given duration
        """
        cur_state = plant_state
        duration = self.controllers_execution_period
        delta_t = self.simulation_sampling
        n_integration_steps = int(duration/delta_t) 
        n_steps_simulated = 0
        while n_steps_simulated < n_integration_steps:
            new_h = cur_state[0] + (cur_state[2] - cur_state[1]) * delta_t
            new_dot_h_own = cur_state[1] + command[0] * delta_t
            new_dot_h_int = cur_state[2] + command[1] * delta_t
            new_tau = cur_state[3] - 1 * delta_t
            cur_state = np.array([new_h, new_dot_h_own, new_dot_h_int, new_tau])
            n_steps_simulated += 1
        return cur_state
        
    def pre_process(self, id_agent, plant_state) -> np.array:
        """Calculate the (non-normalized) neural network input(s) corresponding to the measured state of the plant, for a given agent/controller
        
        Parameters
        ----------
        id_agent : int
            The ID of the agent to be considered
        plant_state : np.array
            The measured state of the plant

        Return
        ------
        1D numpy array
            The (non-normalized) input of the neural network
        """
        if id_agent==0:
            return plant_state
        elif id_agent==1:
            return np.array([-plant_state[0], plant_state[2], plant_state[1], plant_state[3]])
        
    def select_network(self, id_agent, prev_label) -> str:
        """Select the network to be executed by a given agent, depending on its previous label

        Parameters
        ----------
        id_agent : int
            The ID of the agent to be considered
        prev_label : int
            The previous label of the agent

        Return
        ------
        string
            The name of the network to be executed by the agent
        """
        return self.prefix_nnet_names + str(prev_label)

    def normalize_input(self, nnet_name, x) -> np.array:
        """Normalize an input vector x

        Parameters
        ----------
        nnet_name : string
            The name of the network of interest
        x : 1D numpy array
            The input vector to be normalized

        Return
        ------
        1D numpy array
            The normalized input
            
        """
        norm_params = self.norm_parameters[nnet_name]
        x_mean = norm_params[2]
        x_range = norm_params[3]
        x_norm = (x - x_mean) / x_range
        return x_norm
        
    def post_process(self, id_agent, y) -> int:
        """Determine the discrete label corresponding to the output of a neural network, for a given agent
        
        Parameters
        ----------
        id_agent : int
            The ID of the agent to be considered
        y : 1D numpy array
            The output of the neural network

        Return
        ------
        list
            The label(s) corresponding to the output of the neural network
        """
        return np.argmax(y)

class ConcreteFactoryVCAS2(AbstractFactorySystem):
    def createSystem(self, config_system) -> ConcreteProductSystemVCAS2:
        return ConcreteProductSystemVCAS2(config_system)
