#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
   This file created input tgff file for the HEFT algorithm.
"""
__author__ = "Quynh Nguyen, Aleksandra Knezevic, Pradipta Ghosh and Bhaskar Krishnamachari"
__copyright__ = "Copyright (c) 2018, Autonomous Networks Research Group. All rights reserved."
__license__ = "GPL"
__version__ = "2.0"

import re
import os
import create_input
import numpy as np


def init(filename):
    """
    This function read the tgff file and
    build computation matrix, communication matrix, rate matrix.
    TGFF is a useful tool to generate directed acyclic graph, tfgg file represent a task graph.

    Args:
        filename (str): name of output TGFF file

    Returns:
        - int: number of tasks
        - list: task names
        - int: number of processors
        - list: computing matrix
        - list: rate matrix
        - list: file size transfer matrix
        - list: communication matrix
    """

    #NODE_NAMES = os.environ["NODE_NAMES"]
    #node_info = NODE_NAMES.split(":")
    #node_ids = {v:k for k,v in enumerate(node_info)}

    f = open(filename, 'r')
    f.readline()
    f.readline()
    f.readline()

    # Calculate the amount of tasks
    num_of_tasks = 0
    task_names=set()


    # Build a communication matrix
    data = {}
    line = f.readline()
    while line.startswith('\tARC'):
        #line = re.sub(r'\bt\d_', '', line)
        #A = [int(s) for s in line.split() if s.isdigit()]
        l = line.split()
        task_names = task_names.union(set([l[3],l[5]]))
        data[(l[3],l[5])] = l[7]

        line = f.readline()
    num_of_tasks = len(task_names)
    #for line in data:
    #    print line


    while not f.readline().startswith('@computation_cost'):
        pass

    # Calculate the amount of processors
    line = f.readline()
    processor_names = [int(pi.strip('node')) for pi in line.split()[3:]]
    num_of_processors = len(processor_names)
    #print 'Number of processors = %d' % num_of_processors

    # Build a computation matrix
    comp_cost = {}
    line = f.readline()
    while line.startswith('  '):
        _cost = (list(map(int, line.split()[-num_of_processors:])))
        comp_cost[line.split()[0]] = _cost
        line = f.readline()
    #for line in comp_cost:
    #    print line

    """
    Build a rate matrix
    rate = [[1 for i in range(num_of_processors)] for i in range(num_of_processors)]
    for i in range(num_of_processors):
        rate[i][i] = 0
    """
    # Build a network profile matrix
    quadratic_profile = dict()
    while not f.readline().startswith('@quadratic'):
        pass
    line = f.readline()
    #print(line)
    line = f.readline()

    while line.startswith('  '):
        info = line.strip().split()
        k = (int(info[0].strip('node')),int(info[1].strip('node')))
        a,b,c = [float(s) for s in info[2:]]
        quadratic_profile[k]= tuple([a,b,c])
        line = f.readline()

    print('==================')
    # print(quadratic_profile)
    assert num_of_processors == max(processor_names)+1
    # assume processor names: 0, 1, 2...
    return [num_of_tasks, task_names, num_of_processors, processor_names, comp_cost, data, quadratic_profile]
