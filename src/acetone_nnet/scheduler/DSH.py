"""Scheduling heuristics for DSH scheduler.

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

from sys import float_info

from .Gantt import DuplicationList, Gantt
from .Graph import Graph, Node
from .ISH import assign_node as assign_node_ish


def update_ready_queue(ready_queue:list[Node], node:Node) -> None:
    """Update the ready queue with the children of `node` that have no more parents to process."""
    for child in node.children:
        child.parents_to_process -= 1

        if child.parents_to_process == 0:
            ready_queue.append(child)
            ready_queue.sort(key=lambda n: n.level,reverse=True )


def copy_LIP(
        node:Node,
        proc:int,
        gantt:Gantt,
        graph:Graph,
        dupli_list:DuplicationList,
        upper_bond:float = float_info.max,
) -> bool:
    """Attempt to copy the Longest Idle Period (LIP) of a node to a processor's Gantt chart.

    This function recursively tries to insert the LIP into idle times available in
    the Gantt chart while respecting constraints like the maximum execution time (upper bound).

    :param node: The node/task to be copied.
    :type node: Node
    :param proc: The identifier for the processor where the task should be copied.
    :type proc: Int
    :param gantt: The Gantt chart object representing the schedule for all processors.
    :type gantt: Gantt
    :param graph: The data structure representing the task graph dependencies.
    :type graph: Graph
    :param dupli_list: A list managing duplicate tasks across processors.
    :type dupli_list: DuplicationList
    :param upper_bond: Optional; the maximum allowable execution time for the node in this copy attempt.
                       Defaults to the maximum floating-point value.
    :type upper_bond: Float
    :return: True if the LIP was successfully copied to the processor's schedule, False otherwise.
    :rtype: Bool
    """
    lip, node_start_time = gantt.search_dupli_list_lip(node, proc, graph, dupli_list)
    if lip is None:
        return False
    lip_lip, lip_start_time = gantt.get_LIP_and_start_time(lip, proc, graph, dupli_list)

    if not gantt.schedules[proc].in_idle_time(lip, lip_start_time):
        return False

    upper_bond = min(node_start_time, upper_bond)
    while lip_start_time + lip.wcet <= upper_bond:
        inserted = dupli_list.insert(lip, lip_start_time)
        if inserted:
            removed_nodes = dupli_list.shift_task(lip)
            for removed in removed_nodes:
                copy_LIP(removed, proc, gantt, graph, dupli_list)
            return True
        lip_start_time = dupli_list.next_hole(lip_start_time)

    return copy_LIP(lip, proc, gantt, graph, dupli_list, upper_bond=upper_bond)




def TDP(
        node:Node,
        proc:int,
        gantt:Gantt,
        graph:Graph,
) -> tuple[float, DuplicationList]:
    """Calculate the best starting time for a specific node.

    Manages duplication of the Least Immediate Predecessor (LIP)
    on a processor schedule to improve node scheduling.

    This function performs iterative exploration to determine if a duplication of the
    LIP can reduce communication delays, leading to a more efficient scheduling
    time for the specified node on the chosen processor.

    :param node: The computational task node to be scheduled.
    :type node: Node
    :param proc: The identifier of the processor where the node is to be scheduled.
    :type proc: int
    :param gantt: The Gantt chart object managing scheduling information.
    :type gantt: Gantt
    :param graph: The task dependency graph containing relationships and properties.
    :type graph: Graph
    :return: Returns a tuple containing the best starting time for the node
        and the list tracking duplication efforts of LIPs.
    :rtype: tuple[float, DuplicationList]
    """
    schedule = gantt.schedules[proc]
    dupli_list = DuplicationList()
    best_start_time = float_info.max

    # Obtenir LIP et le temps initial
    success = True
    while success and best_start_time > schedule.get_ready_time():
        lip, start_time = gantt.search_dupli_list_lip(node, proc, graph, dupli_list)

        # Si pas de communication, pas besoin de duplication
        if lip is None or gantt.schedules[proc].is_node_scheduled(lip) >= 0 or node.wcet == 0:
            return start_time, dupli_list

        # S'il y a une fenêtre d'inactivité entre ready time et message ready time
        if best_start_time > schedule.get_ready_time():
            if lip not in dupli_list.get_nodes():
            # Tenter de dupliquer le LIP
                success = copy_LIP(node, proc, gantt, graph, dupli_list, upper_bond=best_start_time)
            else:
                success = copy_LIP(lip, proc, gantt, graph, dupli_list)
            if success:
                # Recalcule le start_time avec duplication effective
                _, best_start_time = gantt.search_dupli_list_lip(node, proc, graph, dupli_list)
            else:
                best_start_time = start_time
    return best_start_time, dupli_list

def locate_pe(
        node:Node,
        gantt:Gantt,
        graph:Graph,
) -> tuple[int, float, DuplicationList]:
    """Locate the optimal processing element for a given node.

    The method attempts to minimize the start time
    for the task while considering dependencies and duplications.

    :param node: The task/node to be assigned to a processing element.
    :type node: Node
    :param gantt: An object representing the Gantt chart, including scheduling details
        and processor availability.
    :type gantt: Gantt
    :param graph: The dependency graph for tasks, containing task relationships
        and communication times.
    :type graph: Graph
    :return: A tuple containing the index of the best processor (int), the minimized
        start time for the task (float), and the list of duplications applied
        (DuplicationList).
    :rtype: tuple[int, float, DuplicationList]
    """
    best_proc = None
    best_time = float_info.max
    best_duplication = DuplicationList()

    for proc in range(gantt.nb_proc):
        start_time, dupli_list = TDP(node, proc, gantt, graph)
        if start_time < best_time:
            best_time = start_time
            best_proc = proc
            best_duplication = dupli_list

    return best_proc, best_time, best_duplication

def assign_node(
        node:Node,
        start_time:float,
        gantt:Gantt,
        proc:int,
        list_duplications:DuplicationList,
        ready_queue:list[Node],
        graph:Graph,
) -> None:
    """Assign a node to a processor and manages its scheduling in the Gantt chart.

    Handles duplications processing before the assignment and ensures
    correct integration into the ready queue and graph.

    :param node: The node to be assigned to a processor.
    :type node: Node
    :param start_time: The starting time for node execution.
    :type start_time: float
    :param gantt: The Gantt chart containing all processor schedules.
    :type gantt: Gantt
    :param proc: The processor index to which the node is assigned.
    :type proc: int
    :param list_duplications: List of duplications to be processed before node assignment.
    :type list_duplications: DuplicationList
    :param ready_queue: List of nodes available for processing.
    :type ready_queue: list[Node]
    :param graph: The graph structure representing the task dependencies.
    :type graph: Graph
    :return: None
    """
    schedule = gantt.schedules[proc]
    for (duplicated, duplicated_start_time) in list_duplications.dupli_list:
        schedule.add_node(duplicated, duplicated_start_time)

    assign_node_ish(node,start_time,gantt,proc,ready_queue, graph)

