This work is based on the following paper:

>  Arthur Clavière, Laura Altieri Sambartolomé, Eric Asselin, Christophe Garion, ans Claire Pagetti, "[Verification of machine learning based cyber-physical systems: a comparative study](https://doi.org/10.1145/3501710.3519540)," International Conference on Hybrid Systems: Computation and Control (HSCC), May 2022, Pages 1–16.


This is a simple Python code that allows simulating three systems. All these three systems are cyber-physical systems which combine a _plant_ with one or several _neural network based controller(s)_.

# Systems description

The available systems are described below. Note that a more detailed description can be found at https://dl.acm.org/doi/abs/10.1145/3501710.3519540.


* **ACAS Xu** (airborne collision avoidance system):
  * **plant**: it is composed of two aircraft, called the _ownship_ and the _intruder_. These two aircraft evolve in a horizontal plane. The state of the two aircraft is described by the vector $x_p = (\Delta x, \Delta y, \psi_\text{own}, \psi_\text{int}, v_\text{own}, v_\text{int})$ wherein $(\Delta x, \Delta y)$ are the 2D cartesian coordinates of the intruder relative to the ownship, $\psi_\text{own}$ and $\psi_\text{int}$ are the heading directions of the ownship and the intruder respectively, and $v_\text{own}$ and $v_\text{int}$ are the velocities of the ownship and the intruder respectively.
  * **controller**: while the intruder is assumed to have a uniform rectilinear displacement, the ownship is equipped with a controller that periodically provides a horizontal maneuver advisory to avoid a collision between the two aircraft. This advisory can be either _Clear-of-conflict_ (COC), _Weak left_ (WL), _Weak right_ (WR), _Strong left_ (SL) or _Strong right_ (SR). The controller has an internal state $x_c$, being the latest advisory computed. To determine the new advisory, the controller measures the vector $(\rho, \theta, \phi, v_\text{own}, v_\text{int})$ wherein $\rho$ is the distance between the two aircraft, $\theta$ is the angle of intruder relative to the ownship heading direction and $\phi$ is the heading of the intruder relative to the ownship heading direction. Then, the controller disposes of a collection of neural networks. It chooses one of these networks based on its internal state _i.e.,_ the latest advisory computed. The selected network is executed and, based the the scores produced by the network, a new advisory is produced.
  * Note: the neural networks used in the code and located in `src/systems/acasxu/nnets/` were taken from https://github.com/guykatzz/ReluplexCav2017.
  
* **ACAS Xu 2**: this system is identical to the ACAS Xu, except that the intruder is also equipped with a collision avoidance controller. The controller of the two aircraft are assumed to be executed simultaneously. They are considered as a single controller with internal state $x_c = (x_{c_o}, x_{c_i})$ wherein $x_{c_o}$ and $x_{c_i}$ are the internal states of the ownship and the intruder respectively.
  * Note: the neural networks used in the code and located in `src/systems/acasxu_2/nnets/` were taken from https://github.com/guykatzz/ReluplexCav2017.

* **VCAS 2**: this system is quite similar to the ACAS Xu 2, but the aircraft evolve in a vertical plane instead of evolving in a horizontal plane.

  * **plant**: it is composed of an _ownship_ and an _intruder_, evolving in a vertical plane. The state of the two aircraft is described by the vector $x_p = (h, \dot{h}_\text{own}, \dot{h}_\text{int}, \tau)$ wherein $h$ is the altitude of the intruder relative to the ownship, $dot{h}_\text{own}$ and $dot{h}_\text{int}$ are the vertical velocities of the ownship and the intruder respectively, and $\tau$ is the time before loss of horizontal separation between the two aircraft (in s).
  * **controller**: both the ownship and the intruder are equipped with a collision avoidance controller. The controller of the ownship periodically provides a vertical maneuver advisory, among $9$ possible advisories. As for the ACAS Xu, this controller has an internal state $x_{c_o}$ being the latest advisory computed. To determine the new advisory, the controller measures the vector $(h, \dot{h}_\text{own}, \dot{h}_\text{int}, \tau)$. Then, the controller disposes of a collection of neural networks. It chooses one of these networks based on its internal state _i.e.,_ the latest advisory computed. The selected network is executed and, based the the scores produced by the network, a new advisory is produced. The controller of the intruder behaves similarly. As for the ACAS Xu 2, the two controllers are assumed to be executed simultaneously and they are considered as a single controller, with internal state $x_c = (x_{c_o}, x_{c_i})$.
  * Note: the neural networks used in the code and located in `src/systems/vcas/nnets/` were taken from https://github.com/sisl/VerticalCAS.


# Code organization

The code is located in the `src/` directory. 

* `simulation.py` contains the simulation loop;
* the `systems/` directory contains a class describing each system _i.e.,_ the dynamics of the plant and the operations performed by the controller(s) (_e.g.,_ `system_acasxu.py`)

# Running a set of simulations

For each system, a `.csv` file containing a list of initial states is available in the `init_states/` directory (_e.g._, `init_states_acasxu.csv`). In these files, an initial state is defined as (1) a duration for the simulation (in s), (2) an initial state for the plant $x_{p_0}$ and (3) and initial internal state for the controller $x_{c_0}$.

A simulation of these initial states can be performed by running the script `src/simulate.sh`, in which the system of interest can be chosen. This will generate simulation traces, located in the `simulation_traces/` directory.


# Authors

The Python code on which this is based was originally written by Arthur Clavière and is distributed (see https://svn.onera.fr/schedmcore/branches/) under an LGPL version 3 license, so this directory also has an LGPL LICENSE, which is non-standard compared to other LF code. The pre-trained neural networks were developed at Stanford University as part of the Reluplex project and are released under the Creative Commons Attribution 4.0 International License. The authors of the neural nets are listed in the AUTHORS file.