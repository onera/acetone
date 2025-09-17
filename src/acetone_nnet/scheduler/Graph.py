from abc import ABC
from pathlib import Path

from typing_extensions import Self


class Node(ABC):

    def __init__(self, tag, wcet):
        self.tag = tag
        self.wcet = wcet
        self.level = -1
        self.parents = []
        self.children = []
        self.parents_to_process = 0
        self.is_duplication = []

        if wcet < 0:
            raise TypeError("wcet cannot be negative")

    def copy(self):
        new_node = Node(self.tag, self.wcet)
        new_node.level = self.level
        new_node.parents = self.parents.copy()
        new_node.children = self.children.copy()
        new_node.parents_to_process = len(self.parents)
        return new_node

    def set_parent(self, parent:Self)-> None:
        self.parents.append(parent)
        self.parents_to_process += 1
        parent.children.append(self)

    def set_child(self, child:Self) -> None:
        self.children.append(child)
        child.parents.append(self)
        child.parents_to_process += 1

    def is_predecessor(self, node:Self)->bool:
        for child in self.children:
            if child == node:
                return True
            else:
                return child.is_predecessor(node)
        return False

    def __str__(self):
        string = f" Node: {self.tag}, wcet: {self.wcet}, level:{self.level},  parents:{[parent.tag for parent in self.parents]}, children: {[child.tag for child in self.children]}"
        return string

    def __eq__(self, other):
        if self.tag != other.tag:
            return False
        if self.wcet != other.wcet:
            return False
        if self.level != other.level:
            return False
        if self.parents != other.parents:
            return False
        if self.children != other.children:
            return False
        return True


class Edge(ABC):

    def __init__(self, src: Node, dst: Node, weight:int) -> None:
        self.src = src
        self.dst = dst
        self.weight = weight

        if not isinstance(self.src, Node):
            raise TypeError("Source node must be of type Node")
        if not isinstance(self.dst, Node):
            raise TypeError("Destination node must be of type Node")
        if self.weight < 0:
            raise ValueError("Weight must be positive")

    def __str__(self):
        string = f" Edge: {self.src.tag} --{self.weight}--> {self.dst.tag}"
        return string


class Graph(ABC):

    def __init__(self) -> None:
        self.nodes = []
        self.edges = []
        self.leaf : Node | None = None
        self.roots : list[Node] = []
        self.dir = Path(__file__).parent.resolve()

    def get_edges_from(self, node:Node):
        if node not in self.nodes:
            raise ValueError(f"{node} not in the graph")

        return [e for e in self.edges if e.src == node]

    def get_edges_to(self, node: Node):
        if node not in self.nodes:
            raise ValueError(f"{node} not in the graph")

        return [e for e in self.edges if e.dst == node]

    def get_edge(self, src:Node, dst:Node):
        for e in self.edges:
            if e.src == src and e.dst == dst:
                return e
        raise ValueError(f"No edges between {src} and {dst}")

    def get_node_from_tag(self, tag:str|int):
        for n in self.nodes:
            if n.tag == tag:
                return n
        raise ValueError(f"No node with tag {tag} in the graph")

    def fuse_leaves(self, new_leaf_name="Fused_leaf"):
        leafs = []
        for n in self.nodes:
            if len(n.children) == 0:
                leafs.append(n)

        if len(leafs) > 1:
            new_leaf = Node(new_leaf_name, 0)
            self.add_node(new_leaf)
            for l in leafs:
                self.add_edge(Edge(l, new_leaf, 0))

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def compute_level(self):
        queue = [self.leaf]
        while queue:
            current_node = queue.pop(0)
            queue.extend(current_node.parents)
            queue = [queue[i] for i in range(len(queue)) if queue[i] not in queue[:i]]
            max_children_level = 0
            for child in current_node.children:
                max_children_level = max(max_children_level, child.level)
            current_node.level = max_children_level + current_node.wcet

        for node in self.nodes:
            if node.level < 0:
                return False
        return True

    def reset(self):
        for node in self.nodes:
            node.parents_to_process = len(node.parents)
        self.validate()


    def add_nodes(self, nodes: list[Node]) -> None:
        self.nodes.extend(nodes)

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
        edge.src.set_child(edge.dst)

    def add_edges(self, edges: list[Edge]) -> None:
        self.edges.extend(edges)
        for edge in edges:
            edge.src.set_child(edge.dst)

    def get_communication(self, src:Node,dst:Node):
        com = 0
        for e in self.edges:
            if e.src == src and e.dst == dst:
                com = max(e.weight, com)
        return com

    def is_dag(self):
        for node in self.nodes:
            if node.is_predecessor(node):
                return False
        return True


    def validate(self):
        nb_leaf = 0
        for node in self.nodes:
            if not node.children:
                nb_leaf += 1
                self.leaf = node
            if not node.parents:
                self.roots.append(node)

        if nb_leaf > 1:
            raise ValueError(f"Graph can only have one leaf node ({nb_leaf} > 1)")

        if not self.is_dag():
            raise ValueError("Graph is not a DAG")


    def __str__(self):
        string = "Nodes : \n"
        for node in self.nodes:
            string += "    " + str(node) + "\n"
        string += "\nEdges : \n"
        for edge in self.edges:
            string += "    " + str(edge) + "\n"
        string += f"\nRoot nodes : {[node.tag for node in self.roots]}\n"
        string += f"Leaf node : {self.leaf.tag}"

        return string

    def get_uniproc_finish_time(self):
        finish_time = 0
        for node in self.nodes:
            finish_time += node.wcet
        return finish_time
