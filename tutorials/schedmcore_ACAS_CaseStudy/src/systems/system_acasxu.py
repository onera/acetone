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
from ..utils import *

class ConcreteProductSystemACASXU(AbstractProductSystem):
    """A class corresponding to the 2D ACASXU system and the associated simulation options, implementing the AbstractProductSystem class

    Attributes (Specific)
    ----------
    norm_parameters
        A dictionary recording the normalization parameters for each network composing the NNCS
    prefix_nnet_names
        The prefix of the nnet names
    available_commands
        The list of the commands which can be applied to the plant
    range_ownship_sensors
        The range of ownship sensors, in ft
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
        self.prefix_nnet_names = 'nnet_acas_'
        # The list of the available commands (to be applied to the plant)
        self.available_commands = [0.0,1.5,-1.5,3.5,-3.5] # [Clear-of-conflict, Weak left, Weak right, Strong left, Strong Right], measured in degree, counter clockwise
        # The range of ownship sensors
        self.range_ownship_sensors = 100000.0
        # The collision radius
        self.collision_radius = 500.0
        
    # utility function
    def wrap_to_pi(self, angle):
        """Wrap angle in radians to [-pi,pi]

        Parameters
        ----------
        angle : float
            The angle to be wrapped to [-pi,pi], in rad

        Return
        ------
        float
            The corresponding angle in [-pi,pi], in rad
        """
        # angle % (2 * pi) wrap the angle to [0, 2 * pi]
        # we use this result to wrap the angle to [-pi, pi]
        return (angle + pi) % (2 * pi) - pi
        
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
        for i in range(1,6):
            # retrieve the path to the i^th network
            nnet_filename = 'ACASXU_experimental_v2a_{}_1.nnet'.format(i)
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
        return (np.sqrt(plant_state[0]**2 + plant_state[1]**2) >= self.collision_radius)
        
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
        turn_rate_rad = []
        for prev_label in prev_labels:
            turn_rate_rad.append(self.available_commands[prev_label] * pi / 180.0)
        return np.array(turn_rate_rad)
        
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
            new_delta_x = cur_state[0] + ( - cur_state[5] * np.sin(cur_state[3]) + cur_state[4] * np.sin(cur_state[2])) * delta_t
            new_delta_y = cur_state[1] + ( cur_state[5] * np.cos(cur_state[3]) - cur_state[4] * np.cos(cur_state[2])) * delta_t
            new_psi_own = cur_state[2] + command[0] * delta_t
            new_psi_int = cur_state[3]
            new_v_own = cur_state[4]
            new_v_int = cur_state[5]
            cur_state = np.array([new_delta_x, new_delta_y, new_psi_own, new_psi_int, new_v_own, new_v_int])
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
        # retrieve state variables
        delta_x = plant_state[0]
        delta_y = plant_state[1]
        psi_own = plant_state[2]
        psi_int = plant_state[3]
        v_own = plant_state[4]
        v_int = plant_state[5]
        # 1) compute rho
        rho = np.sqrt(delta_x**2 + delta_y**2)
        # 2) compute theta
        if delta_y > 0:
            angle = -np.arctan(delta_x/delta_y)
        elif delta_x < 0:
            angle = pi/2 + np.arctan(delta_y/delta_x)
        elif delta_x > 0:
            angle = -pi/2 + np.arctan(delta_y/delta_x)
        elif delta_x == 0:
            angle = 0 if delta_y==0 else pi
        theta = angle - psi_own
        # wrap theta into [-pi,pi]
        while theta < -pi:
            theta += 2*pi
        while theta > pi:
            theta -= 2*pi
        # 3) compute psi
        psi = psi_int - psi_own
        # wrap psi into [-pi,pi]
        while psi < -pi:
            psi += 2*pi
        while psi > pi:
            psi -= 2*pi
        return np.array([rho, theta, psi, v_own, v_int])
        
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
        return np.argmin(y)
        
class ConcreteFactoryACASXU(AbstractFactorySystem):
    def createSystem(self, config_system) -> ConcreteProductSystemACASXU:
        return ConcreteProductSystemACASXU(config_system)
