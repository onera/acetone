"""Scheduling heuristics for ISH scheduler.

*******************************************************************************
* ACETONE: Predictable programming framework for ML applications in safety-critical systems
* Copyright (c) 2022. ONERA
* This file is part of ACETONE
*
* ACETONE is free software ;
* you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation ;
* either version 3 of  the License, or (at your option) any later version.
*
* ACETONE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
* without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
* See the GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License along with this program ;
* if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
******************************************************************************
"""

from .Gantt import Gantt, IdleTimeSlot
from .Graph import Graph, Node


def update_ready_queue(ready_queue:list[Node], node:Node) -> None:
    """Update the ready queue with the children of `node` that have no more parents to process."""
    for child in node.children:
        child.parents_to_process -= 1

        if child.parents_to_process == 0:
            ready_queue.append(child)
            ready_queue.sort(key=lambda n: n.level,reverse=True )

def assign_node(
        node:Node,
        start_time:float,
        gantt:Gantt,
        proc:int,
        ready_queue:list[Node],
        graph:Graph,
) -> None:
    """Assign a node to a processor and manages its scheduling in the Gantt chart.

    Handles insertion processing before the assignment and ensures
    correct integration into the ready queue and graph.

    :param node: The node to be assigned to a processor.
    :type node: Node
    :param start_time: The starting time for node execution.
    :type start_time: float
    :param gantt: The Gantt chart containing all processor schedules.
    :type gantt: Gantt
    :param proc: The processor index to which the node is assigned.
    :type proc: int
    :param ready_queue: List of nodes available for processing.
    :type ready_queue: list[Node]
    :param graph: The graph structure representing the task dependencies.
    :type graph: Graph
    :return: None
    """
    schedule = gantt.schedules[proc]
    idle_time_slot = IdleTimeSlot(schedule.get_ready_time(), start_time)
    found = True
    while found and idle_time_slot.get_idle_time() > 0 and ready_queue:
        found = False
        ready_queue_pos = 0
        hole_node = ready_queue[ready_queue_pos]
        while not found and ready_queue_pos < len(ready_queue):
            _, hole_node_start_time = gantt.get_LIP_and_start_time(hole_node,proc,graph)
            found, slot_idx = idle_time_slot.in_slot(start_time, start_time+hole_node.wcet)
            if not found:
                ready_queue_pos += 1
        if found:
            idle_time_slot.add_busy_time(slot_idx, start_time, start_time+hole_node.wcet)
            ready_queue.pop(ready_queue_pos)
            update_ready_queue(ready_queue, hole_node)
            idle_time_slot -= hole_node.wcet

    schedule.add_node(node,start_time)


def locate_pe(
        node:Node,
        gantt:Gantt,
        graph:Graph,
) -> tuple[int, float]:
    """Locate the best processor for executing the specified node in a task graph.

    :param node: The current computational task represented as a node in the graph.
    :type node: Node
    :param gantt: The Gantt chart instance that manages schedules and processors' states.
    :type gantt: Gantt
    :param graph: The task graph representing all tasks and their dependencies.
    :type graph: Graph
    :return: A tuple containing the selected processor ID and the start time for the task.
    :rtype: tuple[int, float]
    """
    proc_first_ready = gantt.first_ready()
    final_proc = proc_first_ready
    start_time = gantt.schedules[final_proc].get_ready_time()

    if node.parents:
        _, start_time = gantt.get_LIP_and_start_time(node,proc_first_ready,graph)

    for parent in node.parents:
        for (proc, _ ) in gantt.locate_node(parent):
            _, new_start_time = gantt.get_LIP_and_start_time(node, proc, graph)
            if new_start_time <= start_time:
                start_time = new_start_time
                final_proc = proc

    return final_proc, start_time


