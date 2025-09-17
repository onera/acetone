from .Gantt import Gantt, IdleTimeSlot
from .Graph import Graph, Node


def update_ready_queue(ready_queue, node):
    for child in node.children:
        child.parents_to_process -= 1

        if child.parents_to_process == 0:
            ready_queue.append(child)
            ready_queue.sort(key=lambda n: n.level,reverse=True )

def assign_node(node:Node, start_time:float, gantt:Gantt, proc:int, ready_queue:list[Node], graph:Graph):
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


def locate_pe(node:Node, gantt:Gantt, graph:Graph):
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


