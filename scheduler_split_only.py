import networkx as nx
import numpy as np
import networkx.drawing
import matplotlib.pyplot as plt
import argparse
import utils
import numpy as np
from scheduler_general import *
from scheduler_dup import *
from scheduler_split import *
from gen_proc_graph import get_proc_graph_spec
import time
from rand_task_gen import get_task_dag
#np.random.seed(6)

from os.path import expanduser
import sys
sys.path.insert(0,'{}/Documents/'.format(expanduser('~')))
import zython.db_util as db

import subprocess
import yaml

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
    parser.add_argument("--db_name",type=str, default="./logs/split_only/default.db")
    parser.add_argument("--max_itr",type=int, default=10,help='max number of iterations allowed in split')
    parser.add_argument("--num_reserve",type=int,default=1,help='number of processors reserved in each cluster in initial mapping')
    parser.add_argument("--dag_path_plot",type=str, default = "./plots/split_only/task_dag.png")

    return parser.parse_args()



if __name__ == '__main__':
    """
    TODO: should throughput_dup and throughput_split be modified to e.g.max{throughput_dup, throughput_init}??
    Therefore, we need further modify throughput_split_avg and throughput_dup_avgs
    """
    args = parse_args()
    #task_names, processor_names, sys_profile = utils.parse_prof_tgff(args.task_path) #genrate task_dag from tgff
    throughput_init = np.zeros(args.num_run)
    throughput_init_re = np.zeros(args.num_run)
    throughput_split = np.zeros(args.num_run)
    improvement = np.zeros(args.num_run)
    inter_clu_thp = np.zeros(args.num_run)
    comp_thp = np.zeros(args.num_run)
    fail_stat = {Scheduler_Dup.FAIL_CANDY: 0,
                 Scheduler_Dup.FAIL_ROUTE: 0,
                 Scheduler_Dup.FAIL_CLUSIZE: 0,
                 Scheduler_Dup.FAIL_INDGR: 0}
    for i in range(args.num_run):
        #g_task  = gen_task_dag(task_names,sys_profile) #generate task_dag from tgff
        g_task = get_task_dag(args.task_yaml,args.dag_path_plot) #generate random tasks
        task_nodes = g_task.nodes(data=True)
        find_initial = False
        while True:
            """
            get initial mapping for the task_dag
            """
            ret = get_proc_graph_spec(args.cluster_yaml,task_nodes)
            #import pdb; pdb.set_trace()
            if ret is None:
                continue
            g_cluster, g_processor, proc_generator = ret
            #print('\n\n bandwidth{}'.format( g_processor.edges.data()))
            proc_generator.visualize()
            scheduler_inst1 = Scheduler_split(g_task.copy(),g_processor.copy(),args.k,args.m,args.spdup,args.max_itr,0)
            scheduler_inst2 = Scheduler_split(g_task,g_processor,args.k,args.m,args.spdup,args.max_itr,args.num_reserve)
            #def __init__(self, g_task, g_processor,k,m,spdup): k: dup at most k*num_of_link_in_task_dag number of links; m:dup when the link throughput < m*sys_throughput,
            if scheduler_inst1.initial_mapping() and scheduler_inst2.initial_mapping():
                #initial_mapping() will update self.sys_throughput
                proc_generator.visualize(sup_exec_g=scheduler_inst2.g_task2proc,viz_outf='./plots/split_only/init.png')
                print('initial throughput = {:5.3f}'.format(scheduler_inst2.sys_throughput))
                find_initial = True

                break
        throughput_init[i] = scheduler_inst1.sys_throughput
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
        inter_clu_thp[i] = np.mean(_inter_clu_thp)
        comp_thp[i] = np.mean(_comp_thp)
        throughput_init_re[i] = scheduler_inst2.sys_throughput
        #initial throughput with processor reservation

        itr, throughput_split[i] = scheduler_inst2.split()
        print('throughput after {} iterations split = {:5.3f}'.format(itr,scheduler_inst2.sys_throughput))
        improvement[i] = (throughput_split[i] - throughput_init[i])/throughput_init[i]
        #improvement means the benefit from split with processor-reservation compared to initial mapping wihtout processor-reservation
        proc_generator.visualize(sup_exec_g=scheduler_inst2.g_task2proc,viz_outf='./plots/split_only/split.png')
    throughput_init_avg = np.average(throughput_init)
    throughput_init_avg_re = np.average(throughput_init_re)
    throughput_split_avg = np.average(throughput_split)
    improvement = np.clip(improvement,0,None)
    #only keep positive numbers in improvement, because it is a static algorithm and we keep the initial mapping without processor reservation if dup and split do not help
    improvement_avg = np.average(improvement)
    #improvement_avg is the throughput improvement brought by dup and split with processor reservation per args.num_resrve
    success_imprv = sum(n>0 for n in improvement)
    success_imprv_rate = success_imprv/args.num_run
    inter_clu_thp_avg = np.average(inter_clu_thp)
    comp_thp_avg = np.average(comp_thp)
    print("the average throughput improvement of split from {} runs is {}%\n. Succeed to improve throughput {} runs out of {} runs with a ratio of {} ".format(args.num_run, improvement_avg*100,success_imprv,args.num_run,success_imprv_rate))

    # ---read parameters in task_yaml and cluster_yaml---
    with open(args.task_yaml) as t_config:
        task_config = yaml.load(t_config)
    with open(args.cluster_yaml) as p_config:
        proc_config = yaml.load(p_config)
        field_x = proc_config['cluster']['field_x']     # width of the field
        field_y = proc_config['cluster']['field_y']           # height of the field
        num_cluster = proc_config['cluster']['num_cluster']         # number of clusters to be placed in the field
        min_distance = proc_config['cluster']['min_distance']       # min distance between adjacent clusters
        num_obstacle = proc_config['cluster']['num_obstacle']         # number of obstacles to be placed in the field
        r_obs_mu = proc_config['cluster']['r_obs_mu']           # mean radius of each obstacle (assume an obstacle is a circle where clusters cannot be placed)
        r_obs_sigma = proc_config['cluster']['r_obs_sigma']        # variance of radius of the obstacles


        clu_size = proc_config['processor']['clu_size']           # each cluster contains <clu_size> number of processors
        r_cluster = proc_config['processor']['r_cluster']           # radius of a cluster. all processors within a cluster should fall in the circle of radius <r_cluster>
        comp_mu  = proc_config['processor']['comp_mu']            # mean value of computation capacity of a processor
        comp_sigma = proc_config['processor']['comp_sigma']
    print(fail_stat)
    # ---------
    # log performance
    # --------
    attr_name = ['git_rev','throughput_init','throughput_init_reserve','throughput_split','improvement(%)','success_imprv_rate','k','m','spdup','depth','width_min','width_max','ccr','workload','workload_sigma','link_data_sigma','deg_mu','deg_sigma',\
    'field_x','field_y','num_cluster','min_distance','num_obstacle','r_obs_mu','r_obs_sigma','clu_size','r_cluster','comp_mu','comp_sigma','split_max_itr','num_reserve','inter_clu_thp','comp_thp','N0_dbm','min_snr_db']
    attr_type = ['TEXT','REAL','REAL','REAL','REAL','REAL','REAL','REAL','REAL','INTEGER','INTEGER','INTEGER','REAL','REAL','REAL','REAL','REAL','REAL',   'REAL','REAL','INTEGER','REAL','INTEGER','REAL','REAL',\
    'INTEGER','REAL','REAL','REAL','INTEGER','INTEGER','REAL','REAL','REAL','REAL']
    d_tuple = (git_rev,throughput_init_avg,throughput_init_avg_re,throughput_split_avg,improvement_avg*100,success_imprv_rate,args.k,args.m,args.spdup,task_config['depth'],task_config['width_min'],task_config['width_max'],task_config['ccr'],\
    task_config['comp_mu'],task_config['comp_sigma'], task_config['link_comm_sigma'],task_config['deg_mu'],task_config['deg_sigma'], field_x,field_y,num_cluster,min_distance,num_obstacle,\
    r_obs_mu,r_obs_sigma,clu_size,r_cluster,comp_mu,comp_sigma,args.max_itr,args.num_reserve,inter_clu_thp_avg,comp_thp_avg,N0,min_snr_db)
    db.basic.populate_db(attr_name,attr_type,*d_tuple,append_time=True,db_name=args.db_name,db_path='.')
