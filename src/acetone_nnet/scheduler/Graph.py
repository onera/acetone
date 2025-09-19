"""Graph representation for the scheduler.

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

from abc import ABC
from pathlib import Path

from typing_extensions import Self


class Node(ABC):
    """Node of the graph."""

    def __init__(self:Self, tag:str|int, wcet:float) -> None:
        """Initialize the node."""
        self.tag = tag
        self.wcet = wcet
        self.level = -1
        self.parents = []
        self.children = []
        self.parents_to_process = 0
        self.is_duplication = []

        if wcet < 0:
            raise TypeError("wcet cannot be negative")

    def copy(self:Self) -> Self:
        """Return a copy of the node."""
        new_node = Node(self.tag, self.wcet)
        new_node.level = self.level
        new_node.parents = self.parents.copy()
        new_node.children = self.children.copy()
        new_node.parents_to_process = len(self.parents)
        return new_node

    def set_parent(self:Self, parent:Self)-> None:
        """Set the parent of the node."""
        self.parents.append(parent)
        self.parents_to_process += 1
        parent.children.append(self)

    def set_child(self:Self, child:Self) -> None:
        """Set the child of the node."""
        self.children.append(child)
        child.parents.append(self)
        child.parents_to_process += 1

    def is_predecessor(self:Self, node:Self)->bool:
        """Return True if the node is a predecessor of the given node."""
        for child in self.children:
            if child == node:
                return True
            return child.is_predecessor(node)
        return False

    def __str__(self:Self) -> str:
        """Return a string representation of the node."""
        return (f" Node: {self.tag}, wcet: {self.wcet}, level:{self.level},  "
                f"parents:{[parent.tag for parent in self.parents]}, "
                f"children: {[child.tag for child in self.children]}"
        )

    def __eq__(self:Self, other:object) -> bool:
        """Return True if the node is equal to the other node."""
        if not isinstance(other, Node):
            return False
        if self.tag != other.tag:
            return False
        if self.wcet != other.wcet:
            return False
        if self.level != other.level:
            return False
        if self.parents != other.parents:
            return False
        return self.children != other.children


class Edge(ABC):
    """Edge of the graph."""

    def __init__(self:Self, src:Node, dst:Node, weight:int) -> None:
        """Initialize the edge."""
        self.src = src
        self.dst = dst
        self.weight = weight

        if not isinstance(self.src, Node):
            raise TypeError("Source node must be of type Node")
        if not isinstance(self.dst, Node):
            raise TypeError("Destination node must be of type Node")
        if self.weight < 0:
            raise ValueError("Weight must be positive")

    def __str__(self:Self) -> str:
        """Return a string representation of the edge."""
        string = f" Edge: {self.src.tag} --{self.weight}--> {self.dst.tag}"
        return string


class Graph(ABC):
    """Graph representation for the scheduler."""

    def __init__(self:Self) -> None:
        """Initialize the graph."""
        self.nodes = []
        self.edges = []
        self.leaf : Node | None = None
        self.roots : list[Node] = []
        self.dir = Path(__file__).parent.resolve()

    def get_edges_from(self:Self, node:Node) -> list[Edge]:
        """Return the edges starting from the given node."""
        if node not in self.nodes:
            raise ValueError(f"{node} not in the graph")

        return [e for e in self.edges if e.src == node]

    def get_edges_to(self:Self, node: Node) -> list[Edge]:
        """Return the edges ending at the given node."""
        if node not in self.nodes:
            raise ValueError(f"{node} not in the graph")

        return [e for e in self.edges if e.dst == node]

    def get_edge(self:Self, src:Node, dst:Node) -> Edge:
        """Return the edge between the given nodes."""
        for e in self.edges:
            if e.src == src and e.dst == dst:
                return e
        raise ValueError(f"No edges between {src} and {dst}")

    def get_node_from_tag(self:Self, tag:str|int) -> Node:
        """Return the node with the given tag."""
        for n in self.nodes:
            if n.tag == tag:
                return n
        raise ValueError(f"No node with tag {tag} in the graph")

    def fuse_leaves(self:Self, new_leaf_name:str|int="Fused_leaf") -> None:
        """Fuse the leaves of the graph into one node with the given name."""
        leafs = []
        for n in self.nodes:
            if len(n.children) == 0:
                leafs.append(n)

        if len(leafs) > 1:
            new_leaf = Node(new_leaf_name, 0)
            self.add_node(new_leaf)
            for l in leafs:
                self.add_edge(Edge(l, new_leaf, 0))

    def add_node(self:Self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def add_nodes(self:Self, nodes: list[Node]) -> None:
        """Add a list of nodes to the graph."""
        self.nodes.extend(nodes)

    def add_edge(self:Self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        edge.src.set_child(edge.dst)

    def add_edges(self:Self, edges: list[Edge]) -> None:
        """Add a list of edges to the graph."""
        self.edges.extend(edges)
        for edge in edges:
            edge.src.set_child(edge.dst)

    def compute_level(self:Self) -> bool:
        """Compute the level of each node in the graph."""
        queue = [self.leaf]
        while queue:
            current_node = queue.pop(0)
            queue.extend(current_node.parents)
            queue = [queue[i] for i in range(len(queue)) if queue[i] not in queue[:i]]
            max_children_level = 0
            for child in current_node.children:
                max_children_level = max(max_children_level, child.level)
            current_node.level = max_children_level + current_node.wcet

        return all(node.level >= 0 for node in self.nodes)

    def reset(self:Self) -> None:
        """Reset the graph."""
        for node in self.nodes:
            node.parents_to_process = len(node.parents)
        self.validate()

    def get_communication(self:Self, src:Node,dst:Node) -> int:
        """Return the communication latency between the given nodes."""
        com = 0
        for e in self.edges:
            if e.src == src and e.dst == dst:
                com = max(e.weight, com)
        return com

    def is_dag(self:Self) -> bool:
        """Return True if the graph is a DAG."""
        return all(not node.is_predecessor(node) for node in self.nodes)


    def validate(self:Self) -> None:
        """Check that the graph is valide (i.e. a DAG with only one leaf)."""
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


    def __str__(self:Self) -> str:
        """Return a string representation of the graph."""
        string = "Nodes : \n"
        for node in self.nodes:
            string += "    " + str(node) + "\n"
        string += "\nEdges : \n"
        for edge in self.edges:
            string += "    " + str(edge) + "\n"
        string += f"\nRoot nodes : {[node.tag for node in self.roots]}\n"
        string += f"Leaf node : {self.leaf.tag}"

        return string

    def get_uniproc_finish_time(self:Self) -> int:
        """Return the finish time of the graph if it was scheduled on mono-core."""
        finish_time = 0
        for node in self.nodes:
            finish_time += node.wcet
        return finish_time
