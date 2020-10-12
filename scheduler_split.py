import networkx as nx
import numpy as np
import networkx.drawing
import copy

from scheduler_dup import *

PRINT = False

class Scheduler_split(Scheduler_Dup):
    FAIL_CANDY = 'no candy found'
    FAIL_ROUTE = 'no route found'
    FAIL_CLUSIZE = 'too many links'
    FAIL_INDGR = 'input cluster cannot be dupped'


    def __init__(self,g_task,g_processor,k,m,spdup,max_itr,num_reserve):
        super().__init__(g_task,g_processor,k,m,spdup,num_reserve)
        self._relay_count = 0
        self._code_failure = None
        self.max_itr = max_itr


    def split_task(self,taskid):
        _g_task2proc = copy.deepcopy(self.g_task2proc)
        _g_proc2task = copy.deepcopy(self.g_proc2task)

        #checked data structure here, no problem

        _procid = _g_task2proc.nodes[taskid]['proc_l'][0]
        
        _avai_procs = self.g_cluster.nodes[_procid[0]]['avai_procs']
        if _avai_procs == []:
            if PRINT:
                print("cannot split task {} because there is no available procs in same cluster".format(taskid))
            return None,None,None,None,None
        _avai_procs_cmp = [self.g_processor.nodes[(_procid[0],_candi_proc)]['comp'][taskid.split('#')[-1]] for _candi_proc in _avai_procs]
    
        _cmp_min = min(_avai_procs_cmp)
        #choose processor with minimum computation time for the task that is split
        _proc_split = _avai_procs[_avai_procs_cmp.index(_cmp_min)]
        _proc_split = (_procid[0],_proc_split)
        _g_task2proc.nodes[taskid]['proc_l'].append(_proc_split)
        _g_task2proc.nodes[taskid]['lambda_l'] = [1.0/len(_g_task2proc.nodes[taskid]['proc_l'])] * len(_g_task2proc.nodes[taskid]['proc_l'])

        self.gen_proc2task(_g_task2proc,_g_proc2task)

        for t,d in _g_task2proc.nodes(data=True):
            for p in d['proc_l']:
                assert _g_proc2task.nodes[p]['task'] == t

        _thp_cur, _component_min, _ = self.calc_overall_throughput(_g_task2proc,_g_proc2task)
        return _g_task2proc,_g_proc2task,_thp_cur,_component_min,_proc_split


    def split_link(self,linkid):
        _g_task2proc,_g_proc2task,_thp_cur,_component_min,_proc_split = self.split_task(linkid[0])
        if _thp_cur is not None and _thp_cur > self.sys_throughput:
            return _g_task2proc,_g_proc2task,_thp_cur,_component_min,_proc_split
        return self.split_task(linkid[1])


    def split(self):
        """
        the overall flow of splitting
        """
        itr = 0

        g_task2proc_debug = copy.deepcopy(self.g_task2proc)
        g_proc2task_debug = copy.deepcopy(self.g_proc2task)
        _, component_min, thp_l = self.calc_overall_throughput(None,None)
        component_debug = component_min

        #---------assume the proc_l and lambda_l both of length 1 ---------------
        for nn in list(g_task2proc_debug.nodes):
            p_l = g_task2proc_debug.nodes[nn]['proc_l']
            l_l = g_task2proc_debug.nodes[nn]['lambda_l']
            assert len(p_l)==1 and len(l_l)==1
        #-----------------------------------------------------------------------

        if type(component_min[0]) is tuple:
            g_task2proc,g_proc2task,thp_cur,component_min,proc_split = self.split_link(component_min[0])
        else:
            g_task2proc,g_proc2task,thp_cur,component_min,proc_split = self.split_task(component_min[0])
        if g_task2proc is None:
            if PRINT:
                print('fail to split in the first iteration')
            return itr, self.sys_throughput


        #---------assume the proc_l and lambda_l both of length 1 ---------------
        for nn in list(g_task2proc_debug.nodes):
            p_l = g_task2proc_debug.nodes[nn]['proc_l']
            l_l = g_task2proc_debug.nodes[nn]['lambda_l']
            assert len(p_l)==1 and len(l_l)==1
        #-----------------------------------------------------------------------



        if thp_cur <= self.sys_throughput:
            if PRINT:
                print('fail to split in the first iteration for throughput after split is {},which is less\
            than before {}'.format(thp_cur,self.sys_throughput))
            return itr, self.sys_throughput

        while (thp_cur > self.sys_throughput and itr < self.max_itr):
            self.g_task2proc = g_task2proc
            self.g_proc2task = g_proc2task
            self.sys_throughput = thp_cur
            self.g_cluster.nodes[proc_split[0]]['avai_procs'].remove(proc_split[1])
            itr += 1
            for t,d in self.g_task2proc.nodes(data=True):
                for p in d['proc_l']:
                    assert self.g_proc2task.nodes[p]['task'] == t
 
            if PRINT:
                print("succeed to split in the {} iteration. Throughput is {}".format(itr,self.sys_throughput))
            # the first successful splitting will be printed  as "the 1 iteration"
            #import pdb; pdb.set_trace()
            if type(component_min[0]) is tuple:
                g_task2proc,g_proc2task,thp_cur,component_min,proc_split = self.split_link(component_min[0])
            else:
                g_task2proc,g_proc2task,thp_cur,component_min,proc_split = self.split_task(component_min[0])
            if g_task2proc is None:
                break

        return itr, self.sys_throughput
