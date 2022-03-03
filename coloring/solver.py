#!/usr/bin/python
# -*- coding: utf-8 -*-
import threading
from multiprocessing import Queue

from line_profiler_pycharm import profile
from collections import defaultdict
from typing import List, Union
from operator import attrgetter
from copy import deepcopy, copy
import numpy as np
import time



class Node:

    def __init__(self, idx, ub: int, node_count: int):
        self.idx = idx
        self.depth = 99999
        self.node_count = node_count
        self.ub = ub
        self.best_obj = self.ub
        self.lb = 0
        self.degree = 0
        self.color = None

        self.color_domains = None
        self.color_not_in = None

        self.lower_bound = None
        self.neighbours = []
        self.constraints = []

        self.next_node: Union[None, Node] = None

    def add_edge(self, neighbour):
        self.neighbours.append(neighbour)
        self.degree += 1

    @property
    @profile
    def get_lower_bound(self):
        # max_lb = max([self.color_domains[n][0] for n in range(self.node_count)]) + 1
        # return max_lb

        # domains = [[color for color in range(self.lb, self.best_obj) if color not in self.color_not_in[n]] for n in range(self.node_count)]

        domains = [list(range(min(self.color_not_in[n].union([1])))) for n in range(self.node_count)]
        max_lb = max([domains[n][0] for n in range(self.node_count) if len(domains[n]) > 0]) + 1

        return max(max_lb, self.lb)


class SearchTree:
    def __init__(self, root, cc):
        self.root = root
        self.cc = cc
        self.best_obj = 9999
        self.best_solution = None

        self.tic = time.time()

        self.evaluation_count = 0
        self.max_evaluation = 2000000

    def reset_color(self, node: Node):
        node.color = None
        node.depth = 99999
        node.ub = node.best_obj
        node.lb = 0


        while node.next_node is not None:
            node.next_node.color = None
            node.next_node.depth = 99999
            node.next_node.ub = node.next_node.best_obj
            node.next_node.lb = 0

            next_node = node.next_node
            node.next_node = None
            node = next_node


    @profile
    def try_all(self, node):

        self.evaluation_count += 1
        if self.evaluation_count % 10000 == 0:
            print(f'tried {self.evaluation_count} values @ {int(time.time() - self.tic)} secs')
            pass
        if self.evaluation_count > 500000:
            return

        # for i in node.color_domains[node.idx]:
        for i in [color for color in range(node.lb, node.ub) if color not in node.color_not_in[node.idx]]:

            # reset colors
            if node.next_node is not None:
                self.reset_color(node.next_node)

            node.color = i

            if i + 1 >= self.best_obj:
                # impossible to improve further, prune
                return

            lower_bound = node.get_lower_bound

            if lower_bound >= self.best_obj:
                # impossible to improve further, prune
                return

            # if node.next_node is not None:
            if node.depth < node.node_count - 1:

                # min_domain = min([len(node.color_domains[n]) for n in range(node.node_count)
                #                       if self.cc.nodes[n].depth > node.depth])
                # where = [n for n in range(node.node_count)
                #          if self.cc.nodes[n].depth > node.depth and len(node.color_domains[n])==min_domain]

                max_not_in = max([len(node.color_not_in[n]) for n in range(node.node_count)
                                      if self.cc.nodes[n].depth > node.depth])
                where = [n for n in range(node.node_count)
                         if self.cc.nodes[n].depth > node.depth and len(node.color_not_in[n])==max_not_in]

                max_degree_idx = np.argmax([self.cc.nodes[n].degree for n in where])
                next_node_idx = where[max_degree_idx]
                node.next_node = self.cc.nodes[next_node_idx]

                # next_domains = [domain.copy() for domain in node.color_domains]
                next_color_not_in = [not_in.copy() for not_in in node.color_not_in]

                current_obj = self.cc.obj_value()
                # TODO reduce memory consumption
                next_node_idx = node.next_node.idx
                # next_domains[next_node_idx] = [color for color in next_domains[next_node_idx]
                #                                     if color <= current_obj]
                # next_color_not_in[next_node_idx] = next_color_not_in[next_node_idx].union(range(current_obj + 1, node.ub))
                node.next_node.ub = min(node.next_node.ub, current_obj+1)

                # remove the current value from the neighbouring nodes' domain
                # rm_color = lambda n: next_domains[n] if n not in node.neighbours \
                #     else [color for color in next_domains[n] if color != i]
                # next_domains = [rm_color(n) for n in range(len(next_domains))]
                _ = [next_color_not_in[neighbour].add(i) for neighbour in node.neighbours]


                # next_domains[node.idx] = [i]
                # next_color_not_in[node.idx] = {n for n in range(node.ub) if n != i}
                next_color_not_in[node.idx] = {i+1}
                # node.ub = i+1
                node.lb = i


                node.next_node.depth = node.depth + 1
                node.next_node.ub = min(node.ub + 1, node.next_node.ub)
                # node.next_node.color_domains = next_domains
                node.next_node.color_not_in = next_color_not_in

                self.try_all(node.next_node)
            else:
                # reach the last node, check obj value
                if self.cc.obj_value() < self.best_obj:
                    self.best_obj = self.cc.obj_value()
                    self.best_solution = ' '.join([str(node.color) for node in self.cc.nodes])
                    print(self.cc.obj_value())
                    print(self.best_solution)
                    print('New Best')

                    # prune all color domains
                    for nodenode in self.cc.nodes:
                        for n in range(nodenode.node_count):
                            nodenode.ub = min(self.best_obj, nodenode.ub)
                            nodenode.best_obj = self.best_obj
                            # nodenode.color_domains[n] = [color for color in nodenode.color_domains[n]
                            #                                if color < self.best_obj]


class ConstraintsChecker:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def obj_value(self):
        return max([node.color for node in self.nodes if node.color is not None]) + 1

def solve_it(input_data):
    threading.stack_size(200000000)
    result = Queue()
    thread = threading.Thread(target=solve_it_thread, args=(input_data, result))
    thread.start()

    thread.join()
    # pool = ThreadPool(processes=1)
    # results = pool.apply_async(target=solve_it_thread, args=(input_data))

    return result.get()


def solve_it_thread(input_data, result):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    nodes = [Node(idx=i, ub=node_count, node_count=node_count) for i in range(node_count)]

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

        nodes[int(parts[0])].add_edge(int(parts[1]))
        nodes[int(parts[1])].add_edge(int(parts[0]))

    max_degree = 0
    min_degree = 999999
    for node in nodes:
        if node.degree > max_degree:
            max_degree = node.degree
        if node.degree < min_degree:
            min_degree = node.degree

    cc = ConstraintsChecker(nodes=nodes, edges=edges)
    nodes_sorted = sorted(nodes, key=attrgetter('degree'), reverse=True)

    nodes_sorted[0].color_not_in = [set() for n in range(node_count)]
    # nodes_sorted[0].color_not_in[nodes_sorted[0].idx] = nodes_sorted[0].color_not_in[nodes_sorted[0].idx].union([n for n in range(1, node_count)])


    # nodes_sorted[0].color_domains = [[i for i in range(node_count)] for n in range(node_count)]
    # nodes_sorted[0].color_domains[nodes_sorted[0].idx] = [0]
    nodes_sorted[0].depth = 0
    nodes_sorted[0].ub = nodes_sorted[0].depth+1

    st = SearchTree(root=nodes_sorted[0], cc=cc)
    st.try_all(st.root)


    # prepare the solution in the specified output format
    output_data = str(st.best_obj) + ' ' + str(0) + '\n'
    output_data += st.best_solution

    result.put(output_data)
    return output_data



if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
