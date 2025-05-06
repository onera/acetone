'''
This file is part of a program that allows simulating the ACAS Xu and VCAS neural network controlled systems
Copyright (c) 2023 Arthur Claviere ONERA/ISAE-SUPAERO

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
'''

from abc import ABC, abstractmethod
import numpy as np
import subprocess
import os
import shutil

# local imports

class AbstractProductSystem(ABC):
    """An abstract class corresponding to a neural network contolled system (NNCS) and the associated simulation options
    
    Attributes (Generic)
    ----------
    path_nnets
        The path to the neural networks of the different agents/controllers composing the NNCS
    plant_state_dim
        The dimension of the state of the plant
    n_agents
        The number of agents/controllers composing the NNCS
    controllers_execution_period
        The execution period of the agents/controllers
    simulation_sampling
        The sampling period to be used for the simulation of the dynamics with Euler's method
    decision_criteria
        The decision criteria for post processing, in {'min', 'max'}
    
    Methods (Generic)
    -------
    get_path_nnets()
        Get the path to the neural networks of the different agents/controllers composing the NNCS
    get_plant_state_dim()
        Get the dimension of the state of the plant
    get_n_agents()
        Get the number of agents/controllers composing the NNCS
    get_controllers_execution_period()
        Get the execution period of the agents/controllers
    get_simulation_sampling()
        Get the sampling period to be used for the simulation of the dynamics with Euler's method
    get_decision_criteria()
        Get the decision criteria for post processing, in {'min', 'max'}
        
    Methods
    -------
    get_prefix_nnet_names()
        Get the prefix for the nnet_names
    read_networks()
        Read the neural networks of all the agents/controllers composing the NNCS
    is_safe_state(plant_state)
        Determine whether the state of the plant is safe
    prev_labels_to_commands(prev_labels)
        Convert the previous labels of the agents/controllers into a set of commands for the plant
    simulate_dynamics_euler(plant_state, command)
        Simulate the dynamics using Euler's method, for the execution period of the controller
    pre_process(id_agent, plant_state)
        Calculate the (non-normalized) neural network input(s) corresponding to the measured state of the plant, for a given agent/controller
    select_network(id_agent, prev_label)
        Select the network to be executed by a given agent, depending on its previous label 
    normalize_input(nnet_name, x)
        Normalize an input vector
    post_process(id_agent, y)
        Determine the discrete label corresponding to the output of a neural network, for a given agent
    """
    def __init__(self, config_system):
        """Constructor"""
        self.path_nnets = config_system.get('Path_nnets')
        self.plant_state_dim = config_system.getint('Plant_state_dim')
        self.n_agents = config_system.getint('N_agents')
        self.controllers_execution_period = config_system.getfloat('Controllers_execution_period')
        self.simulation_sampling = config_system.getfloat('Simulation_sampling')
        self.decision_criteria = config_system.get('Decision_criteria')
        
    def get_path_nnets(self) -> str:
        """Get the path to the neural networks of the different agents/controllers composing the NNCS

        Parameters
        ----------
        None

        Return
        ------
        string
            The path to the neural networks of the different agents/controllers composing the NNCS

        """
        return self.path_nnets
        
    def get_plant_state_dim(self) -> int:
        """Get the dimension of the state of the plant

        Parameters
        ----------
        None

        Return
        ------
        int
            The dimension of the state of the plant

        """
        return self.plant_state_dim

    def get_n_agents(self) -> int:
        """Get the number of agents/controllers composing the NNCS

        Parameters
        ----------
        None

        Return
        ------
        int
            The number of agents/controllers composing the NNCS
            
        """
        return self.n_agents

    def get_controllers_execution_period(self) -> float:
        """Get the execution period of the agents/controllers

        Parameters
        ----------
        None

        Return
        ------
        float
            The execution period of the agents/controllers
            
        """
        return self.controllers_execution_period
        
    def get_simulation_sampling(self) -> float:
        """Get the sampling period to be used ffor the simulation of the dynamics with Euler's method

        Parameters
        ----------
        None

        Return
        ------
        float
            The the sampling period to be used for the simulation of the dynamics with Euler's method
            
        """
        return self.simulation_sampling
        
    def get_decision_criteria(self) -> str:
        """Get the decision criteria for post processing, in {'min', 'max'}
        
        Parameters
        ----------
        None

        Return
        ------
        string
            The the decision criteria for post processing, in {'min', 'max'}
        
        """
        return self.decision_criteria
                
    @abstractmethod
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
        pass
        
    @abstractmethod
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
        pass
             
    @abstractmethod
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
        pass
        
    @abstractmethod
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
        pass
        
    @abstractmethod
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
        pass
            
    @abstractmethod
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
        pass
            
    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
        
    @abstractmethod
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
        pass
                              
class AbstractFactorySystem(ABC):
    @abstractmethod
    def createSystem(self, config_system) -> AbstractProductSystem:
        pass
