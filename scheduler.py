import networkx as nx
import numpy as np
import networkx.drawing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import utils
import numpy as np
from scheduler_general import *
from scheduler_dup import *
from scheduler_split import *
#   from gen_proc_graph import get_proc_graph_spec
from gen_proc_graph_rvar import get_proc_graph_spec
import time
from rand_task_gen import get_task_dag
import copy,math
#np.random.seed(6)

from os.path import expanduser
import sys
sys.path.insert(0,'{}/Projects/'.format(expanduser('~')))

import subprocess
import yaml
import multiprocessing as mp
from itertools import chain
import time
import json


git_rev = subprocess.Popen("git rev-parse --short HEAD", shell=True, stdout=subprocess.PIPE,universal_newlines=True).communicate()[0]
#git version number

def gen_task_dag(task_names,sys_profile):
    """
    maybe generate the CNAD task graph?
    OUTPUT:
        networkx graph of the task dag
    """
    task_dag = nx.DiGraph()
    task_dag.add_nodes_from(list(task_names))
    for k,v in sys_profile['data'].items():
        task_dag.add_edge(k[0],k[1],data = v)
    nx.draw(task_dag, with_labels=True)
    plt.savefig('task.png')
    return task_dag



def parse_args():
    parser = argparse.ArgumentParser(description = "scheduling algorithm.")
    parser.add_argument("--task_path", type=str, default='./input1.tgff',
        help = "the path of the tgff file containing profiling information.")
    parser.add_argument("--cluster_yaml",type=str, default='./cluster_config/proc_config.yml')
    parser.add_argument("--num_run",type=int, default=100)
    parser.add_argument("--k",type=float, default=0.3,help="dup at most k*num_of_link_in_task_dag number of links")
    parser.add_argument("--m",type=float, default=1.5,help="dup when the link throughput < m*sys_throughput")
    parser.add_argument("--spdup",type=float, default=1.03,help="pick links in find_candidates and find_routes with such spdup")
    parser.add_argument("--task_yaml",type=str, default="./taskconfig/task_config.yml")
    parser.add_argument("--db_name",type=str, default="./logs/split_and_dup/default.db")
    parser.add_argument("--max_itr",type=int, default=20,help='max number of iterations allowed in split')
    parser.add_argument("--num_reserve",type=int,default=0,help='number of processors reserved in each cluster in initial mapping')
    parser.add_argument("--dag_path_plot",type=str, default = "./plots/dup_and_split/task_dag.png")
    parser.add_argument("--num_threads",type=int, default=8)
    parser.add_argument("--debug",action='store_true',help='if in debug mode, do serial execution (no multi-processing)')
    return parser.parse_args()


def run_mp(args,num_run,tid,output_stat,debug=False):
    stat = {'initial reserve': [],
            'initial no-reserve': [],
            'inter clu thp': [],
            'comp thp': [],
            'fail code': dict(),
            'dup reserve': [],
            'split reserve': [],
            'improvement dup-split': [],
            'improvement dup': []}
    for i in range(num_run):
        t1 = time.time()
        print('run {}_{}'.format(tid,i))
        g_task = get_task_dag(args.task_yaml,None) #generate random tasks
        task_nodes = g_task.nodes(data=True)
        find_initial = False
        while True:
            """
            get initial mapping for the task_dag
            """
            ret = get_proc_graph_spec(args.cluster_yaml,task_nodes)
            if ret is None:
                continue
            g_cluster, g_processor, proc_generator = ret
            if debug:
                proc_generator.visualize()
            scheduler_inst1 = Scheduler_split(copy.deepcopy(g_task),copy.deepcopy(g_processor),args.k,args.m,args.spdup,args.max_itr,0)
            scheduler_inst2 = Scheduler_split(g_task,g_processor,args.k,args.m,args.spdup,args.max_itr,args.num_reserve)
            if scheduler_inst1.initial_mapping() and scheduler_inst2.initial_mapping():
                #initial_mapping() will update self.sys_throughput
                #----------assert the length of proc_l and lambda_l are of length 1 after initial mapping-----
                for nn in list(scheduler_inst2.g_task2proc.nodes):
                    p_l = scheduler_inst2.g_task2proc.nodes[nn]['proc_l']
                    l_l = scheduler_inst2.g_task2proc.nodes[nn]['lambda_l']
                    assert len(p_l)==1 and len(l_l)==1
                #----------------------------------------------------------------------------------------------
                if debug:
                    proc_generator.visualize(sup_exec_g=scheduler_inst2.g_task2proc,viz_outf='./plots/dup_and_split/init.png')
                #print('initial throughput = {:5.3f}'.format(scheduler_inst2.sys_throughput))
                stat['initial reserve'].append(scheduler_inst2.sys_throughput)
                find_initial = True
                break
        #throughput_init[i] = scheduler_inst1.sys_throughput
        stat['initial no-reserve'].append(scheduler_inst1.sys_throughput)
        #initial throughput without processor reservation
        _,__,throughput_l = scheduler_inst1.calc_overall_throughput()
        _inter_clu_thp = []
        _comp_thp = []
        for k,v in throughput_l.items():
            if type(k[0]) is tuple:
                if k[1][0][0] != k[1][1][0]:
                    _inter_clu_thp.append(v)
            else:
                _comp_thp.append(v)
        stat['inter clu thp'].append(min(_inter_clu_thp))
        stat['comp thp'].append(min(_comp_thp))
        #throughput_init_re[i] = scheduler_inst2.sys_throughput
        #initial throughput with processor reservation
        _thp_cur, _component_min,_ = scheduler_inst2.calc_overall_throughput()
        #the line above is for debug only
        cnt = scheduler_inst2.duplicate()
        #---------assume the proc_l and lambda_l both of length 1 after dup---------------
        for nn in list(scheduler_inst2.g_task2proc.nodes):
            p_l = scheduler_inst2.g_task2proc.nodes[nn]['proc_l']
            l_l = scheduler_inst2.g_task2proc.nodes[nn]['lambda_l']
            assert len(p_l)==1 and len(l_l)==1
        #-----------------------------------------------------------------------
        if scheduler_inst2._code_failure is not None:
            if scheduler_inst2._code_failure in stat['fail code'].keys():
                stat['fail code'][scheduler_inst2._code_failure] += 1
            else:
                stat['fail code'][scheduler_inst2._code_failure] = 1
        #print('throughput after duplication = {:5.3f}'.format(scheduler_inst2.sys_throughput))
        stat['dup reserve'].append(scheduler_inst2.sys_throughput)
        #throughput_dup[i] = scheduler_inst2.sys_throughput

        itr, throughput_split_i = scheduler_inst2.split()
        #print('throughput after {} iterations split = {:5.3f}'.format(itr,scheduler_inst2.sys_throughput))
        stat['split reserve'].append(scheduler_inst2.sys_throughput)
        assert throughput_split_i == stat['split reserve'][i]
        #improvement[i] = (throughput_split[i] - throughput_init[i])/throughput_init[i]
        stat['improvement dup-split'].append((throughput_split_i-stat['initial no-reserve'][i])/stat['initial no-reserve'][i])
        #improvement means the benefit from dup and split with processor-reservation compared to initial mapping wihtout processor-reservation
        #improvement_dup[i] = (throughput_dup[i] - throughput_init[i])/throughput_init[i]
        stat['improvement dup'].append((stat['dup reserve'][i]-stat['initial no-reserve'][i])/stat['initial no-reserve'][i])
        #NOTE:improvement_dup is only meaningful when args.num_reserve=0
        #proc_generator.visualize(sup_exec_g=scheduler_inst2.g_task2proc,viz_outf='./plots/dup_and_split/dup_and_split.png')
        t2 = time.time()
    #print('thread done')
    if debug:
        return stat
    else:
        with open('./thread_stat/stat_{}.json'.format(tid),'w') as f:
            json.dump(stat,f)
        output_stat.put(stat)




if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        args.num_threads = 1
    run_per_thd = math.ceil(args.num_run/args.num_threads)
    args.num_run = run_per_thd*args.num_threads
    #task_names, processor_names, sys_profile = utils.parse_prof_tgff(args.task_path) #genrate task_dag from tgff
    throughput_init = np.zeros(args.num_run)
    throughput_init_re = np.zeros(args.num_run)     # throughput after "reserve"
    throughput_dup = np.zeros(args.num_run)
    throughput_split = np.zeros(args.num_run)
    improvement = np.zeros(args.num_run)
    improvement_dup = np.zeros(args.num_run)
    inter_clu_thp = np.zeros(args.num_run)
    comp_thp = np.zeros(args.num_run)
    fail_stat = {Scheduler_Dup.FAIL_CANDY: 0,
                 Scheduler_Dup.FAIL_ROUTE: 0,
                 Scheduler_Dup.FAIL_CLUSIZE: 0,
                 Scheduler_Dup.FAIL_INDGR: 0}
    print('=====================')
    print('initiating {:5d} runs'.format(args.num_run))
    print('=====================')

    t1 = time.time()
    if not args.debug:
        ret_stat = []
        for ii in range(0,run_per_thd,6):
            run_per_thd_i = min(ii+6,run_per_thd)-ii
            output_stat = mp.Queue()
            processes = [mp.Process(target=run_mp,args=(args,run_per_thd_i,i+ii,output_stat)) for i in range(args.num_threads)]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            ret_stat.extend([output_stat.get() for p in processes])
    else:
        ret_stat = [run_mp(args,run_per_thd,0,None,debug=True)]
    throughput_init = np.array(list(chain.from_iterable([r['initial no-reserve'] for r in ret_stat])))
    throughput_init_re = np.array(list(chain.from_iterable([r['initial reserve'] for r in ret_stat])))
    throughput_dup = np.array(list(chain.from_iterable([r['dup reserve'] for r in ret_stat])))
    throughput_split = np.array(list(chain.from_iterable([r['split reserve'] for r in ret_stat])))
    improvement = np.array(list(chain.from_iterable([r['improvement dup-split'] for r in ret_stat])))
    improvement_dup = np.array(list(chain.from_iterable([r['improvement dup'] for r in ret_stat])))
    for r in ret_stat:
        for k,v in r['fail code'].items():
            fail_stat[k] += v
    t2 = time.time()
    print('==============================================')
    print('Done {:5d} runs in {:7.2f}s ({:7.2f}s per run)'.format(args.num_run,(t2-t1),(t2-t1)/args.num_run))
    print('==============================================')
    throughput_init_avg = np.average(throughput_init)
    throughput_init_avg_re = np.average(throughput_init_re)
    throughput_dup_avg = np.average(throughput_dup)
    throughput_split_avg = np.average(throughput_split)
    improvement = np.clip(improvement,0,None)
    #only keep positive numbers in improvement, because it is a static algorithm and we keep the initial mapping without processor reservation if dup and split do not help
    improvement_avg = np.average(improvement)
    #improvement_avg is the throughput improvement brought by dup and split with processor reservation per args.num_resrve
    success_imprv = sum(n>0 for n in improvement)
    success_imprv_rate = success_imprv/args.num_run
    improvement_dup_avg = np.average(improvement_dup)
    inter_clu_thp_avg = np.average(inter_clu_thp)
    comp_thp_avg = np.average(comp_thp)
    print('             num runs: {}'.format(args.num_threads*run_per_thd))
    print('      improvement dup: {:.2f}%'.format(improvement_dup_avg*100))
    print("improvement dup-split: {:.2f}%".format(improvement_avg*100))
    print('        success ratio: {:.3f}'.format(success_imprv_rate))
    print(fail_stat)

    # ---read parameters in task_yaml and cluster_yaml---
    with open(args.task_yaml) as t_config:
        task_config = yaml.load(t_config)
    with open(args.cluster_yaml) as p_config:
        proc_config = yaml.load(p_config)
        N0 = proc_config['env']['N0']
        eta = proc_config['env']['eta']
        psi = proc_config['env']['psi']
        field_x = proc_config['cluster']['field_x']     # width of the field
        field_y = proc_config['cluster']['field_y']           # height of the field
        num_cluster = proc_config['cluster']['num_cluster']         # number of clusters to be placed in the field
        min_distance = proc_config['cluster']['min_distance']       # min distance between adjacent clusters
        num_obstacle = proc_config['cluster']['num_obstacle']         # number of obstacles to be placed in the field
        r_obs_mu = proc_config['cluster']['r_obs_mu']           # mean radius of each obstacle (assume an obstacle is a circle where clusters cannot be placed)
        r_obs_sigma = proc_config['cluster']['r_obs_sigma']        # variance of radius of the obstacles
        min_snr_db = proc_config['cluster']['min_snr_db']

        clu_size = proc_config['processor']['clu_size']           # each cluster contains <clu_size> number of processors
        r_cluster = proc_config['processor']['r_cluster']           # radius of a cluster. all processors within a cluster should fall in the circle of radius <r_cluster>
        comp_mu  = proc_config['processor']['comp_mu']            # mean value of computation capacity of a processor
        comp_sigma = proc_config['processor']['comp_sigma']

        #-------- for r_var only-------------------------------------------------------------------------
        min_deg = proc_config['cluster']['min_deg']                 #the min connection degree of a cluster
        rate_mu = proc_config['cluster']['rate_mu']         #the mean of inter-cluster data rate
        rate_sigma = proc_config['cluster']['rate_sigma']     #the standard dev of inter-clsuter data rate
        r_neigh_max = proc_config['cluster']['r_neigh_max']        #the max distance btw 2 neighboring clusters
        rate_in_mu = proc_config['processor']['rate_in_mu']     # the mean of intra-cluster data rate
        rate_in_sigma = proc_config['processor']['rate_in_sigma']    #the std of intra-clsuter data rate
        #------------------------------------------------------------------------------------------------



    # ---------
    # log performance
    # --------
    attr_name = ['git_rev','throughput_init','throughput_init_reserve','throughput_dup','throughput_split',\
                 'improvement(%)','improvement_dup(%)(valid when reserve=0)','success_imprv_rate',\
                 'k','m','spdup','depth','width_min','width_max','ccr','workload','workload_sigma','link_data_sigma','deg_mu','deg_sigma',\
                 'field_x','field_y','num_cluster','min_distance','num_obstacle','r_obs_mu','r_obs_sigma','clu_size','r_cluster',\
                 'comp_mu','comp_sigma','split_max_itr','num_reserve','inter_clu_thp','comp_thp',\
                 'N0_dBm','eta','min_snr_db',\
                 'r_neigh_max','clu_min_deg','rate_mu','rate_sigma','rate_in_mu','rate_in_sigma']
    attr_type = ['TEXT','REAL','REAL','REAL','REAL','REAL','REAL','REAL','REAL','REAL','REAL','INTEGER','INTEGER','INTEGER','REAL','REAL','REAL','REAL','REAL','REAL',   'REAL','REAL','INTEGER','REAL','INTEGER','REAL','REAL',\
    'INTEGER','REAL','REAL','REAL','INTEGER','INTEGER','REAL','REAL','REAL','REAL','REAL',\
    'REAL','INTEGER','REAL','REAL','REAL','REAL']
    d_tuple = (git_rev,throughput_init_avg,throughput_init_avg_re,throughput_dup_avg,throughput_split_avg,improvement_avg*100,improvement_dup_avg*100,success_imprv_rate,args.k,args.m,args.spdup,task_config['depth'],task_config['width_min'],task_config['width_max'],task_config['ccr'],\
    task_config['comp_mu'],task_config['comp_sigma'], task_config['link_comm_sigma'],task_config['deg_mu'],task_config['deg_sigma'], field_x,field_y,num_cluster,min_distance,num_obstacle,\
    r_obs_mu,r_obs_sigma,clu_size,r_cluster,comp_mu,comp_sigma,args.max_itr,args.num_reserve,inter_clu_thp_avg,comp_thp_avg,N0,eta,min_snr_db,\
    r_neigh_max,min_deg,rate_mu,rate_sigma,rate_in_mu,rate_in_sigma)
    db.basic.populate_db(attr_name,attr_type,*d_tuple,append_time=True,db_name=args.db_name,db_path='.')
