from .DSH import assign_node as assign_node_dsh
from .DSH import locate_pe as locate_pe_dsh
from .DSH import update_ready_queue as update_ready_queue_dsh
from .Gantt import DuplicationList, Gantt
from .Graph import Graph
from .ISH import assign_node as assign_node_ish
from .ISH import locate_pe as locate_pe_ish
from .ISH import update_ready_queue as update_ready_queue_ish


def init_ready_queue(graph : Graph):
    ready_queue = graph.roots
    ready_queue.sort(key=lambda node: node.level)
    return ready_queue

def graph_assigned(graph:Graph, gantt):
    for node in graph.nodes:
        assigned = False
        for p in range(gantt.nb_proc):
            if gantt.schedules[p].is_node_scheduled(node) >= 0:
                assigned = True
                break
        if not assigned:
            return False
    return True

def scheduler_DSH(graph : Graph, nb_proc : int):
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

def scheduler_ISH(graph : Graph, nb_proc : int):
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

def scheduler(heuristic:str, graph : Graph, nb_proc : int):
    if heuristic.lower() == "dsh":
        return scheduler_DSH(graph, nb_proc)
    if heuristic.lower() == "ish":
        return scheduler_ISH(graph, nb_proc)
    raise ValueError("Unknown heuristic")