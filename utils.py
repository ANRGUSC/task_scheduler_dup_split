import re
import os
import numpy as np


def parse_prof_tgff(filename):
    """
    This function read the tgff file and
    build computation matrix, communication matrix, rate matrix.
    TGFF is a useful tool to generate directed acyclic graph, tfgg file represent a task graph.

    Args:
        filename (str): name of output TGFF file

    Returns:
        - list: task names
        - list: computing matrix
        - list: rate matrix
        - list: file size transfer matrix
        - list: communication matrix
    """
    # profile information is stored in this dict:
    # {'comp': comp_cost, 'data': data, 'link quad': quadatic_profile}
    ret_profile = dict()

    f = open(filename, 'r')
    f.readline()
    f.readline()
    f.readline()

    # Calculate the amount of tasks
    task_names=set()


    # Build a communication dictionary
    data = {}
    line = f.readline()
    while line.startswith('\tARC'):
        l = line.split()
        task_names = task_names.union(set([l[3],l[5]]))
        data[(l[3],l[5])] = float(l[7])
        line = f.readline()
    ret_profile['data'] = data


    while not f.readline().startswith('@computation_cost'):
        pass

    line = f.readline()
    processor_names = [int(pi) for pi in line.split()[3:]]
    num_of_processors = len(processor_names)

    # Build a computation dictionary
    comp_cost = {}
    line = f.readline()
    while line.startswith('  '):
        _cost = (list(map(int, line.split()[-num_of_processors:])))
        comp_cost[line.split()[0]] = _cost
        line = f.readline()
    ret_profile['comp'] = comp_cost

    """
    Build a rate matrix
    rate = [[1 for i in range(num_of_processors)] for i in range(num_of_processors)]
    for i in range(num_of_processors):
        rate[i][i] = 0
    """
    # Build a network profile matrix
    quadatic_profile = dict()
    while not f.readline().startswith('@quadratic'):
        pass
    line = f.readline()
    line = f.readline()

    while line.startswith('  '):
        info = line.strip().split()
        k = (int(info[0].strip('node')),int(info[1].strip('node')))
        a,b,c = [float(s) for s in info[2:]]
        quadatic_profile[k]= tuple([a,b,c])
        line = f.readline()
    ret_profile['link quad'] = quadatic_profile
    # print(quadatic_profile)
    assert num_of_processors == max(processor_names)+1
    return task_names, processor_names, ret_profile
