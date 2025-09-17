from sys import float_info

from .Gantt import DuplicationList, Gantt
from .Graph import Graph, Node
from .ISH import assign_node as assign_node_ish


def update_ready_queue(ready_queue, node):
    for child in node.children:
        child.parents_to_process -= 1

        if child.parents_to_process == 0:
            ready_queue.append(child)
            ready_queue.sort(key=lambda n: n.level,reverse=True )


def copy_LIP(node, proc, gantt, graph, dupli_list, upper_bond = float_info.max):
    """
    Copie récursive du LIP de `node` sur `proc`, mise à jour de dupli_list.
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
        else:
            lip_start_time = dupli_list.next_hole(lip_start_time)

    return copy_LIP(lip, proc, gantt, graph, dupli_list, upper_bond=upper_bond)




def TDP(node, proc, gantt, graph):
    """
    TDP = Task Duplication Process
    Retourne le temps de démarrage optimal de `node` sur `proc`,
    ainsi que la liste de duplications à effectuer (DuplicationList).
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

def locate_pe(node, gantt, graph):
    """
    Choisit le processeur qui permet d'exécuter `node` le plus tôt,
    en tenant compte des duplications possibles.
    Retourne : (index_proc, start_time, duplication_list)
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

def assign_node(node:Node, start_time:float, gantt:Gantt, proc:int, list_duplications:DuplicationList, ready_queue:list[Node], graph:Graph):
    schedule = gantt.schedules[proc]
    for (duplicated, duplicated_start_time) in list_duplications.dupli_list:
        schedule.add_node(duplicated, duplicated_start_time)

    assign_node_ish(node,start_time,gantt,proc,ready_queue, graph)

