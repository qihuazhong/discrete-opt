#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import random
from typing import List

import matplotlib.pyplot as plt
from line_profiler_pycharm import profile


class Point:
    def __init__(self, key, x, y):
        self.key = key
        self.x = x
        self.y = y

        # self.out_distance = 0
        # self.next = None
        # self.last = None
        self.neighbours = []

    def out_neighbour(self, in_neighbour: int) -> int:
        if self.neighbours.index(in_neighbour) == 0:
            return self.neighbours[1]
        elif self.neighbours.index(in_neighbour) == 1:
            return self.neighbours[0]
        else:
            raise Exception


def distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def sample_close_points(pts_idx, pt2_idx, dist_mat, node_count):

    distances = dist_mat[pt2_idx].tolist()
    rand_int = int(np.random.randn() * node_count * 0.1)

    sorted_distances = sorted(distances)
    idx = distances.index(sorted_distances[rand_int])

    i = 1
    while idx in pts_idx + [pt2_idx]:
        idx = distances.index(sorted_distances[rand_int+i])
        i += 1
        # print(idx, pt1_idx, pt2_idx)

    return idx


def closest_point(pt1: Point, other_pts: List[Point], dist_mat):
    other_pts_idx = [pt.key for pt in other_pts]

    # distances = [distance(pt1, other_pt) for other_pt in other_pts]
    min_idx = np.argmin(dist_mat[pt1.key][other_pts_idx])

    return min_idx


def distance_matrix(pts):
    return np.asarray([[distance(p1, p2) for p1 in pts] for p2 in pts])


def tour_distance(solution, points):
    # calculate the length of the tour
    obj = distance(points[solution[-1]], points[solution[0]])
    for index in range(0, len(points)-1):
        obj += distance(points[solution[index]], points[solution[index + 1]])
    return obj


def longest_edge(solution, dist_mat, node_count: int):
    closed_solution = solution + [solution[0]]
    distances = [dist_mat[(closed_solution[i], closed_solution[i + 1])] for i in range(len(solution))]
    rand_int = int(np.random.randn() * node_count * 0.2)

    idx = distances.index(
        sorted(distances)[rand_int])

    return closed_solution[idx], closed_solution[idx+1]


def varify_solution(solution, node_count):
    return len(set(solution)) == node_count


def get_solution(points: List[Point]):
    solution = []
    pt: Point = points[0]
    out_idx = pt.neighbours[0]

    while len(solution) < len(points):
        solution.append(pt.key)
        next_idx = pt.out_neighbour(in_neighbour=out_idx)
        out_idx = pt.key
        pt = points[next_idx]

    return solution


def get_greedy_solution(points: List[Point], dist_mat):
    pts: List[Point] = points.copy()
    next_pt_inx = 0
    starting_pt = pts[next_pt_inx]
    while len(pts) > 0:
        pt1: Point = pts.pop(next_pt_inx)
        if len(pts) > 0:
            next_pt_inx = closest_point(pt1, pts, dist_mat=dist_mat)
            # pt1.out_distance = distance(pt1, pts[next_pt_inx])
            pt1.neighbours.append(pts[next_pt_inx].key)
            pts[next_pt_inx].neighbours.append(pt1.key)
        else:
            # pt1.out_distance = distance(pt1, starting_pt)
            pt1.neighbours.append(starting_pt.key)
            starting_pt.neighbours.append(pt1.key)

    return get_solution(points)


def two_opt(solution, points, trials=30):
    pass


@profile
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    node_count = int(lines[0])

    points = []
    for i in range(1, node_count+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(i-1, float(parts[0]), float(parts[1])))

    # ============================
    # build a trivial solution

    dist_mat = distance_matrix(points)
    solution = get_greedy_solution(points, dist_mat)
    closed_solution = solution + [solution[0]]
    assert varify_solution(solution, node_count)
    print(solution)

    for i in range(1000000):
        # t_1, t_2 = longest_edge(solution, dist_mat, node_count)
        # rand = random.randint(0, node_count-1)
        # t_4 = closed_solution[rand]
        # t_3 = closed_solution[rand+1]

        # t_3 = sample_close_points([t_1], t_2, dist_mat, node_count)  # in
        # t_4 = solution[(solution.index(t_3) - 1) % node_count]  # out

        # assert t_3 not in [t_1, t_2]

        # original distance of two edges
        # ori_distance = dist_mat[(t_1, t_2)] + dist_mat[(t_4, t_3)]
        # new_distance = dist_mat[(t_1, t_4)] + dist_mat[(t_3, t_2)]
        #
        # t_5 = sample_close_points([t_1, t_2, t_3], t_3, dist_mat, node_count)
        # t_6 = solution[(solution.index(t_5) - 1) % node_count]

        # assert t_5 not in [t_1, t_2, t_3, t_4]
        # assert t_6 not in [t_1, t_2, t_3, t_4]

        rand_ints = np.random.choice(range(node_count), 3, replace=False)
        rand_ints.sort()
        t_1, t_2 = closed_solution[rand_ints[0]], closed_solution[rand_ints[0]+1]
        t_3, t_4 = closed_solution[rand_ints[1]], closed_solution[rand_ints[1] + 1]
        t_5, t_6 = closed_solution[rand_ints[2]], closed_solution[rand_ints[2] + 1]

        ori_distance = dist_mat[(t_6, t_5)] + dist_mat[(t_1, t_2)] + dist_mat[(t_4, t_3)]

        new_distance_e = dist_mat[(t_1, t_3)] + dist_mat[(t_6, t_4)] + dist_mat[(t_5, t_2)]
        new_distance_f = dist_mat[(t_1, t_5)] + dist_mat[(t_2, t_4)] + dist_mat[(t_3, t_6)]
        new_distance_g = dist_mat[(t_1, t_4)] + dist_mat[(t_2, t_6)] + dist_mat[(t_3, t_5)]
        new_distance_h = dist_mat[(t_1, t_4)] + dist_mat[(t_2, t_5)] + dist_mat[(t_3, t_6)]


        # if new_distance < ori_distance:
        #     # swap
        #     points[t_1].neighbours.remove(t_2)
        #     points[t_2].neighbours.remove(t_1)
        #
        #     points[t_4].neighbours.remove(t_3)
        #     points[t_3].neighbours.remove(t_4)
        #
        #     points[t_1].neighbours.append(t_4)
        #     points[t_4].neighbours.append(t_1)
        #
        #     points[t_2].neighbours.append(t_3)
        #     points[t_3].neighbours.append(t_2)
        #
        min_new_distance = min(new_distance_e, new_distance_f, new_distance_g, new_distance_h)
        min_delta = min_new_distance - ori_distance
        if min_delta < 0:
            points[t_1].neighbours.remove(t_2)
            points[t_2].neighbours.remove(t_1)

            points[t_4].neighbours.remove(t_3)
            points[t_3].neighbours.remove(t_4)

            points[t_6].neighbours.remove(t_5)
            points[t_5].neighbours.remove(t_6)

            if min_new_distance == new_distance_e:
                points[t_1].neighbours.append(t_3)
                points[t_3].neighbours.append(t_1)

                points[t_6].neighbours.append(t_4)
                points[t_4].neighbours.append(t_6)

                points[t_5].neighbours.append(t_2)
                points[t_2].neighbours.append(t_5)

            elif min_new_distance == new_distance_f:
                points[t_1].neighbours.append(t_5)
                points[t_5].neighbours.append(t_1)

                points[t_2].neighbours.append(t_4)
                points[t_4].neighbours.append(t_2)

                points[t_3].neighbours.append(t_6)
                points[t_6].neighbours.append(t_3)

            elif min_new_distance == new_distance_g:
                points[t_1].neighbours.append(t_4)
                points[t_4].neighbours.append(t_1)

                points[t_6].neighbours.append(t_2)
                points[t_2].neighbours.append(t_6)

                points[t_5].neighbours.append(t_3)
                points[t_3].neighbours.append(t_5)

            elif min_new_distance == new_distance_h:
                points[t_1].neighbours.append(t_4)
                points[t_4].neighbours.append(t_1)

                points[t_6].neighbours.append(t_3)
                points[t_3].neighbours.append(t_6)

                points[t_5].neighbours.append(t_2)
                points[t_2].neighbours.append(t_5)


            solution = get_solution(points)
            closed_solution = solution + [solution[0]]

            # print(t_4, ori_distance, new_distance, new_distance < ori_distance)
            print(f'improved {min_delta:.2f} @ {i}th iteration')
            # print(solution)
            if not varify_solution(solution, node_count):
                print(t_1, t_2, t_3, t_4, t_5, t_6)
                print('failed', solution, len(set(solution)))
                assert False


    # print(solution)

    # ============================



    obj = tour_distance(solution, points)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    x = []
    y = []
    for idx in range(0, node_count+1):
        x.append(points[closed_solution[idx]].x)
        y.append(points[closed_solution[idx]].y)
    plt.plot(x, y)
    plt.show()


    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. '
              '(i.e. python solver.py ./data/tsp_51_1)')
