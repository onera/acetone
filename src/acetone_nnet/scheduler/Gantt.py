from abc import ABC
from sys import float_info

from .Graph import Graph, Node


class IdleTimeSlot:

    def __init__(self,start:float,end:float):
        self.start = start
        self.end = end
        if end <= start:
            self.slots = []
        else:
            self.slots = [[start, end]]

    def in_slot(self, start_time, end_time):
        if end_time < start_time:
            raise ValueError(f"start {start_time} can't  be after end {end_time}")
        for i in range(len(self.slots)):
            begin = self.slots[i][0]
            last = self.slots[i][1]
            if begin <= start_time and end_time <= last:
                return True, i
        return False, None

    def add_busy_time(self,slot_idx, start_time,end_time):
        if end_time < start_time:
            raise ValueError(f"start {start_time} can't  be after end {end_time}")
        slot_start = self.slots[slot_idx][0]
        slot_end = self.slots[slot_idx][1]
        new_slot_left, new_slot_right = [], []

        if start_time != slot_start:
            new_slot_left  = [slot_start, start_time]
        if end_time != slot_end:
            new_slot_right = [end_time, slot_end]

        if new_slot_right:
            self.slots.insert(slot_idx+1,new_slot_right)
        if new_slot_left:
            self.slots.insert(slot_idx+1,new_slot_left)
        self.slots.pop(slot_idx)

    def clear_after(self, time):
        for i in range(len(self.slots)):
            if self.slots[i][0] > time:
                self.slots.pop(i)
        if self.slots:
            self.slots[-1][1] = self.end
        else:
            self.slots = [[self.start, self.end]]

    def get_idle_time(self):
        idle_time = 0
        for slot in self.slots:
            idle_time += slot[1] - slot[0]
        return idle_time

    def __str__(self):
        return str(self.slots)

class DuplicationList(ABC):

    def __init__(self):
        self.dupli_list: list[tuple[Node, float]] = []
        self.nb_dupli = 0

    def __str__(self):
        string =  "Duplication list : "
        for (node, start_time) in self.dupli_list:
            string += "(" + str(node.tag) + " : " + str(start_time) + " -> " + str(start_time + node.wcet) + ")"
        return string


    def get_nodes(self):
        nodes = [tup[0] for tup in self.dupli_list]
        return nodes

    def get_start_time(self, node:Node):
        for (n, st) in self.dupli_list:
            if n == node:
                return st
        return -float_info.max

    def insert(self, node:Node, start_time:float):
        if node in self.get_nodes():
            return False

        to_insert = True
        for duplicated in self.get_nodes():
            if self.get_start_time(duplicated) <= start_time < self.get_start_time(duplicated) + duplicated.wcet:
                to_insert = False
                break

            if start_time <= self.get_start_time(duplicated) < start_time + node.wcet:
                to_insert = False
                break

        if to_insert:
            self.dupli_list.append((node, start_time))
            self.dupli_list.sort(key=lambda tup: tup[1])
            self.nb_dupli += 1
        return to_insert

    def shift_task(self, node: Node):
        start_time = self.get_start_time(node)
        removed_nodes = []
        if start_time  >= 0:
            removed_nodes = [
                    duplicated
                    for duplicated, duplicated_start_time in self.dupli_list
                    if duplicated_start_time > start_time
            ]
            self.dupli_list = [
                    (duplicated, duplicated_start_time)
                    for duplicated, duplicated_start_time in self.dupli_list
                    if duplicated_start_time <= start_time
            ]
        return removed_nodes

    def next_hole(self, initial_start_time:int):
        new_start_time = initial_start_time
        for duplicated, duplicated_start_time in self.dupli_list:
            if duplicated_start_time + duplicated.wcet > new_start_time:
                new_start_time = duplicated_start_time + duplicated.wcet
                break
        return new_start_time

    def get_ready_time(self):
        return self.dupli_list[-1][1] + self.dupli_list[-1][0].wcet if self.dupli_list else -float_info.max

class Schedule(ABC):

    def __init__(self, idx:int):
        self.idx = idx
        self.schedule : list[tuple[Node, float]] = []

    def check(self):
        for i in range(1, len(self.schedule)):
            prev_task = self.schedule[i-1]
            curr_task = self.schedule[i]
            if prev_task[1] <= curr_task[1] < prev_task[1] + prev_task[0].wcet:
                raise ValueError("Node can't be superposed")

            if prev_task in self.schedule[i:]:
                raise ValueError(f"Node {prev_task[0].tag} duplicated in schedule")

    def remove_node(self, node):
        for i in range(len(self.schedule)):
            if self.schedule[i][0] == node:
                self.schedule.pop(i)
                break

    def add_node(self, node, start_time):
        self.schedule.append((node, start_time))
        self.schedule.sort(key=lambda x: (x[1], x[0].wcet))
        self.check()

    def copy(self):
        new_schedule = Schedule(self.idx)
        for node, start_time in self.schedule:
            new_schedule.add_node(node, start_time)
        return new_schedule


    def is_node_scheduled(self, node):
        for tup in self.schedule:
            if node == tup[0]:
                return tup[1]
        return -1

    def get_ready_time(self):
        if self.schedule:
            return self.schedule[-1][1] + self.schedule[-1][0].wcet
        return 0

    def in_idle_time(self, node:Node, start_time:float):
        if self.schedule and start_time + node.wcet <= self.schedule[0][1]:
            return True
        if start_time >= self.get_ready_time():
            return True
        for i in range(1, len(self.schedule)):
            prev_task = self.schedule[i-1]
            curr_task = self.schedule[i]
            if prev_task[1] + prev_task[0].wcet <= start_time and start_time + node.wcet <= curr_task[1]:
                return True
        return False


    def __str__(self):
        string = f"{self.idx} : "
        for tup in self.schedule:
            string += f"| {tup[1]} : {tup[0].tag} |"
        return string


class Gantt(ABC):

    def __init__(self,nb_proc):
        self.schedules:list[Schedule] = [Schedule(i) for i in range(nb_proc)]
        self.nb_proc = nb_proc

    def __str__(self):
        string = "Gantt [\n"
        for i in range(self.nb_proc):
            string += "    " + str(self.schedules[i]) + "\n"
        return string + "]"

    def copy(self):
        new_gantt = Gantt(self.nb_proc)
        for i in range(self.nb_proc):
            new_gantt.schedules[i] = self.schedules[i].copy()
        return new_gantt

    def remove_node(self, node, proc):
        self.schedules[proc].remove_node(node)

    def add_node(self, node, start_time, proc):
        self.schedules[proc].add_node(node, start_time)

    def locate_node(self, node):
        node_proc = []
        for proc in self.schedules:
            st = proc.is_node_scheduled(node)
            if  st >= 0:
                node_proc.append((proc.idx, st))
        return node_proc

    def first_ready(self):
        first_ready = 0
        first_ready_time = float_info.max
        for proc in self.schedules:
            ready_time = proc.get_ready_time()
            if ready_time < first_ready_time:
                first_ready_time = ready_time
                first_ready = proc

        return first_ready.idx

    def get_LIP_and_start_time(self, node: Node, proc:int, graph:Graph, dupli_list:DuplicationList = DuplicationList()):
        schedule = self.schedules[proc]
        # Earliest processor available
        start_time = -1
        LIP = None

        for parent in node.parents:
            # If parent duplicated, takes the corresponding start time
            dupli = dupli_list.get_start_time(parent)
            if dupli >= 0:
                parent_constraints = dupli + parent.wcet
            else: # Otherwise takes the earliest instance amongst the cores
                parent_procs = self.locate_node(parent)
                parent_constraints = float_info.max
                for proc_idx, st in parent_procs:
                    com = 0 if proc_idx == proc else graph.get_communication(parent, node)
                    parent_constraints = min(parent_constraints, st + parent.wcet + com)

            # Retrieve the latest constraint
            if start_time < parent_constraints:
                start_time = parent_constraints
                LIP = parent

        return LIP, max(start_time, schedule.get_ready_time())

    def search_dupli_list_lip(self,node:Node,proc:int,graph:Graph, dupli_list:DuplicationList):
        start_time = self.schedules[proc].get_ready_time()
        schedule = self.schedules[proc].copy()
        lip = None

        for parent in node.parents:
            if parent in dupli_list.get_nodes():
                parent_lip, parent_constraints = self.search_dupli_list_lip(parent,proc,graph,dupli_list)
                parent_constraints = max(
                    parent_constraints + parent.wcet,
                    dupli_list.get_start_time(parent) + parent.wcet,
                    schedule.get_ready_time(),
                )
            else:
                parent_procs = self.locate_node(parent)
                parent_constraints = float_info.max
                parent_lip = parent
                for proc_idx, st in parent_procs:
                    com = 0 if proc_idx == proc else graph.get_communication(parent, node)
                    if st + parent.wcet + com < parent_constraints:
                        parent_constraints = max(
                            st + parent.wcet + com,
                            schedule.get_ready_time(),
                        )#, dupli_list.get_ready_time())

            if start_time < parent_constraints:
                start_time = parent_constraints
                lip = parent_lip
        return lip, start_time

    def validate(self, graph:Graph):
        for proc in self.schedules:
            proc.check()

        for node in graph.nodes:
            node_procs = self.locate_node(node)
            if not node_procs:
                raise ValueError(f"Node {node.tag} not scheduled")

        for node in graph.nodes:
            node_procs = self.locate_node(node)
            if len(node_procs) > 1:
                min_proc = min(node_procs, key=lambda x: x[1])[1]
                node.is_duplication.extend([n_proc for (n_proc, n_st) in node_procs if n_st > min_proc])
            for n_proc, n_st in node_procs:
                for parent in node.parents:
                    is_parent_scheduled = False
                    parent_procs = self.locate_node(parent)
                    for p_proc, p_st in parent_procs:
                        if (p_proc == n_proc and p_st + parent.wcet <= n_st)  or (p_proc != n_proc and p_st + parent.wcet + graph.get_communication(parent, node) <= n_st):
                            is_parent_scheduled = True
                            break
                    if not is_parent_scheduled:
                        raise ValueError(f"Node {node.tag} at time {n_st} scheduled before parent {parent.tag}")
        return True

    def get_finish_time(self):
        end_time = 0
        for schedule in self.schedules:
            end_time = max(schedule.get_ready_time(), end_time)
        return end_time

    def clean(self, graph:Graph):
        """Remove all redundant duplications"""
        non_redundant_nodes = []
        cleaned = False

        for i in range(self.nb_proc):
            for j in range(len(self.schedules[i].schedule)):
                node, start_time = self.schedules[i].schedule[j]
                for parent in node.parents:
                    parent_procs = self.locate_node(parent)
                    if not parent_procs:
                        continue

                    if i in [proc_idx for proc_idx, _ in parent_procs]:
                        non_redundant_nodes.append((parent, i))
                    else:
                        com = graph.get_communication(parent,node)
                        min_idx = None
                        min_start = float_info.max
                        for proc_idx, proc_st in parent_procs:
                            if proc_st + com < min_start:
                                min_start = proc_st + com
                                min_idx = proc_idx
                        non_redundant_nodes.append((parent,min_idx))

                if not node.children and node not in [r[0] for r in non_redundant_nodes]:
                    non_redundant_nodes.append((node,i))

        for i in range(self.nb_proc):
            for node, _ in self.schedules[i].schedule:
                if (node, i) not in non_redundant_nodes:
                    cleaned = True
                    self.remove_node(node, i)

        if cleaned:
            self.clean(graph)
