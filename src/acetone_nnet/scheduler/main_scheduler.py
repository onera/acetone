"""Main scheduling heuristics.

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

from .DSH import assign_node as assign_node_dsh
from .DSH import locate_pe as locate_pe_dsh
from .DSH import update_ready_queue as update_ready_queue_dsh
from .Gantt import DuplicationList, Gantt
from .Graph import Graph, Node
from .ISH import assign_node as assign_node_ish
from .ISH import locate_pe as locate_pe_ish
from .ISH import update_ready_queue as update_ready_queue_ish


def init_ready_queue(graph : Graph) -> list[Node]:
    """Initialize the ready queue."""
    ready_queue = graph.roots
    ready_queue.sort(key=lambda node: node.level)
    return ready_queue

def graph_assigned(graph:Graph, gantt:Gantt) -> bool:
    """Check if all the nodes of the graph are assigned."""
    for node in graph.nodes:
        assigned = False
        for p in range(gantt.nb_proc):
            if gantt.schedules[p].is_node_scheduled(node) >= 0:
                assigned = True
                break
        if not assigned:
            return False
    return True

def scheduler_DSH(graph : Graph, nb_proc : int) -> Gantt:
    """Schedule the graph using the DSH heuristic."""
    graph.compute_level()
    gantt = Gantt(nb_proc)
    ready_queue = init_ready_queue(graph)
    node = ready_queue.pop(0)
    assign_node_dsh(node,0,gantt,0,DuplicationList(),ready_queue, graph)
    while not graph_assigned(graph, gantt):
        update_ready_queue_dsh(ready_queue, node)
        node = ready_queue.pop(0)
        proc, start_time, list_duplication = locate_pe_dsh(node, gantt, graph)
        assign_node_dsh(node,start_time,gantt,proc,list_duplication,ready_queue,graph)
    return gantt

def scheduler_ISH(graph : Graph, nb_proc : int) -> Gantt:
    """Schedule the graph using the ISH heuristic."""
    graph.compute_level()
    gantt = Gantt(nb_proc)
    ready_queue = init_ready_queue(graph)
    node = ready_queue.pop(0)
    assign_node_ish(node,0,gantt,0,ready_queue, graph)
    while not graph_assigned(graph, gantt):
        update_ready_queue_ish(ready_queue, node)
        node = ready_queue.pop(0)
        proc, start_time = locate_pe_ish(node, gantt, graph)
        assign_node_ish(node,start_time,gantt,proc,ready_queue,graph)
    return gantt

def scheduler(heuristic:str, graph : Graph, nb_proc : int) -> Gantt:
    """Schedule the graph using the specified heuristic."""
    if heuristic.lower() == "dsh":
        return scheduler_DSH(graph, nb_proc)
    if heuristic.lower() == "ish":
        return scheduler_ISH(graph, nb_proc)
    raise ValueError("Unknown heuristic")