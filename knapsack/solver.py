#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from operator import attrgetter
from typing import List, Union

import sys
sys.setrecursionlimit(5000000)


Item = namedtuple("Item", ['index', 'value', 'weight', 'value_density'])


class Node(object):
    def __init__(self, item_index: int, x: str = '', value=0.0, remaining_capacity=None):
        self.left = None  # selecting the item
        self.right = None  # not selecting the item

        self.item_index = item_index

        self.x = x  # decision variables
        self.value: float = value
        self.remaining_capacity = remaining_capacity
        self.bound = None


class SearchTree(object):
    def __init__(self, root: Node, items: List[Item]):
        self.root = root
        self.best_obj = 0
        self.best_solution = None
        self.items = items  # Should be sorted by a heuristic
        self.item_count = len(items)

        self.evaluation_count = 0
        self.max_evaluation = 2000000

    def create_left(self, node: Node):
        # check feasibility
        next_index = node.item_index + 1
        if self.items[next_index].weight > node.remaining_capacity:
            # infeasible for a left node
            return None

        if not node.left:
            # create a left nodes
            left_node = Node(item_index=next_index,
                             x=node.x + '1',
                             value=node.value + self.items[next_index].value,
                             remaining_capacity=node.remaining_capacity - self.items[next_index].weight)
            node.left = left_node

        return node.left


    def create_right(self, node: Node):
        next_index = node.item_index + 1

        if not node.right:
            # create a right node
            right_node = Node(item_index=next_index,
                              x=node.x + '0',
                              value=node.value,
                              remaining_capacity=node.remaining_capacity)
            node.right = right_node

        return node.right


    def dfs(self, node: Union[None, Node]):

        if node is None:
            return

        if self.evaluation_count > self.max_evaluation:
            # print('max evaluation reached')
            return

        next_index = node.item_index + 1
        if next_index == self.item_count:
            # bottom of the tree (leaf) has been reached
            if node.value > self.best_obj:
                self.best_obj = node.value
                self.best_solution = node.x
        else:

            # Evaluate the best bound of the current node
            node.bound = self.evaluate_bound(node)
            if node.bound > self.best_obj:
                # Branch, continue exploring
                left_node = self.create_left(node)
                self.dfs(left_node)

                right_node = self.create_right(node)
                self.dfs(right_node)
            else:
                # Otherwise, cut this branch
                pass

        del node

    def evaluate_bound(self, node: Node):
        self.evaluation_count += 1

        value = node.value
        remaining_capacity = node.remaining_capacity

        for item_idx in range(node.item_index+1, self.item_count):

            item = self.items[item_idx]

            if item.weight <= remaining_capacity:
                frac = 1
                value += item.value * frac
                remaining_capacity -= item.weight * frac
            else:
                frac = remaining_capacity / item.weight
                value += item.value * frac
                remaining_capacity -= item.weight * frac
                break

        return value


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1]), float(parts[0]) / float(parts[1])))


    items_sorted = sorted(items, key=attrgetter('value_density'), reverse=True)
    # print(items_sorted)
    root = Node(item_index=-1, remaining_capacity=capacity)
    search_tree = SearchTree(root=root, items=items_sorted)

    search_tree.dfs(root)

    # prepare the solution in the specified output format
    output_data = str(int(search_tree.best_obj)) + ' ' + str(1) + '\n'

    # print(search_tree.best_solution)
    # print(items_sorted)
    # print(search_tree.item_count)

    solution = ['0']*search_tree.item_count

    for idx, item in enumerate(items_sorted):
        solution[item.index] = search_tree.best_solution[idx]

    output_data += ' '.join(solution)

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
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
