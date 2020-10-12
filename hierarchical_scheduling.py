# Hierarchical execution graph

# adj list (np.array())
# super-node dictionary
#       each super-node is represented by a dictionary of processors

import networkx as nx
import numpy as np
import networkx.drawing
import matplotlib.pyplot as plt
import argparse
import utils
import numpy as np
#from gen_testcase import gen_testcase
from gen_proc_graph import get_proc_graph_spec



PATH =  "task_splitting_scheduling/input_0.tgff"

class Scheduler:
    def __init__(self, task_dag, processor_g, cluster_g):
        """
        INPUT:
            task_dag            networkx graph representing dependencies of application tasks
            processor_g         networkx graph representing processor interconnection
                                node attr: an array representing the processing speed of one processor for every task
                                edge attr: the bandwidth between two processors
        """
        self.task_dag = task_dag
        self.processor_g = processor_g
        self.cluster_g = cluster_g
        # ---------------------------------
        # sup_exec_g: super execution graph
        #       networkx graph, whose topology is the same as task_dag
        #       node attr: list of lambda; list of processor id
        # constructed by self._dup()
        # updated by self._split()
        self.sup_exec_g = nx.DiGraph()
        self.sup_exec_g.add_edges_from(self.task_dag.edges())
        import pdb; pdb.set_trace()
        for taskid in list(self.sup_exec_g.nodes()):
            self.sup_exec_g.node[taskid]["in_proc"] = []
            self.sup_exec_g.node[taskid]["weight"] = []


    def _add_mapping(self,procid,taskid,weight):
        self.sup_exec_g.node[taskid]["in_proc"].append(procid)
        self.sup_exec_g.node[taskid]["weight"] = weight



    def _update_cluster_g(self, extra_mapping):
        """
        this function updates the cluster graph by removing avai_procs in the cluster. extra_mapping is in the format of {task:proc}
        """
        for v in extra_mapping.values():
            _cluster = self.processor_g.nodes[v]['cluster']
            self.cluster_g.nodes[_cluster]['avai_procs'].remove(v)



    def _get_throughput(self,component):
        """

        """
        if type(component) is tuple:
            p_suproc = component[0]
            c_suproc = component[1]
            p_procs = np.array(self.sup_exec_g.nodes[p_suproc]["in_proc"])
            p_weight = np.array(self.sup_exec_g.nodes[p_suproc]["weight"])
            p_weight_acc = np.zeros(len(p_procs))
            for i in range(len(p_procs)):
                if i == 0:
                    p_weight_acc[i] = p_weight[i]
                else:
                    p_weight_acc[i] = p_wight_acc[i-1]+p_weight[i]
            c_procs = np.array(self.sup_exec_g.nodes[c_suproc]["in_proc"])
            c_weight = np.array(self.sup_exec_g.nodes[c_suproc]["weight"])
            c_weight_acc = np.zeros(len(c_procs))
            for j in range(len(c_procs)):
                if j ==0:
                    c_weight_acc[j] = c_weight[j]
                else:
                    c_weight_acc[j] = c_weight_acc[j-1]+c_weight[j]
            merge_w = np.unique(np.concatenate((p_weight_acc,c_weight_acc),0))
            merge_w = np.insert(merge_w,0,0)
            p_idx = 0
            c_idx = 0
            in_link = dict()
            for i in range(1,len(merge_w)):
                in_link[(p_procs[p_idx],c_procs[c_idx])] = merge_w[i] - merge_w[i-1]

                if merge_w[i] == p_weight_acc[p_idx]:
                    p_idx= p_idx+1
                if merge_w[i] == c_weight_acc[c_idx]:
                    c_idx = c_idx+1
                #i = i + 1
            _throu = dict()
            for k,v in in_link.items():
                _quadra = self.processor_g.edges[k]["quadratic"]
                _data = self.task_dag.edges[component]["data"]
                _throu[k] = 1/(_quadra[0]*((v*_data)**2)+_quadra[1]*(v*_data)+_quadra[2])
                ###to test non-quadratic case, set _quadra[0]=_quadra[2]=0, _quadra[1]=1/BW
            _key_min = min(_throu.keys(), key = (lambda x:_throu[x]))

        else:

            _throu = dict()
            for i in self.sup_exec_g.nodes[component]['in_proc']:
                _idx = self.sup_exec_g.nodes[component]['in_proc'].index(i)
                _weight = self.sup_exec_g.nodes[component]['weight'][_idx]
                _comp = self.processor_g.nodes[i]['comp'][component]
                if _comp != 0:
                    _throu[i] = 1/(_comp * _weight)
                else:
                    _throu[i] = float('inf')
                    #e.g. i is a relay node with _comp=0

            _key_min = min(_throu.keys(), key = (lambda x: _throu[x]))
        return _throu[_key_min]



    def get_sys_throughput(self):
        component_thp = {}
        min_thp = float('inf')
        bottleneck = ''
        # **** store component_thp in np.array, and get min_thp by np.min; bottleneck (of the node index) by np.argmin ****
        for v in list(self.sup_exec_g.nodes):
            cur_thp = self._get_throughput(v)
            component_thp[v] = cur_thp
            if cur_thp<min_thp:
                min_thp = cur_thp
                bottleneck = v
        # **** same comment as before ****
        for e in list(self.sup_exec_g.edges):
            cur_thp = self._get_throughput(e)
            component_thp[e] = cur_thp
            if cur_thp<min_thp:
                min_thp =  cur_thp
                bottleneck = e
        return min_thp, bottleneck, component_thp
        #NB:make a list of the throughput of every sup_node and sup_link



    def _dup_bfs_parent_cluster(self, mapping, link):
        #this function returns the parent cluster of the source cluster of the bad
        #link
        #it buils a bfs tree with root start_t
        # link is an element in inter_links_dup. It is an edge of task_dag
        g = self.task_dag
        start_t = link[0]
        start_p = mapping[start_t]
        start_c = self.processor_g.nodes[start_p]['cluster']
        frontier = [start_t]
        explored = []
        parent = {}
        leaf_list = []
        while frontier:
            expand = frontier.pop(0)
            explored.append(expand)
            if self.processor_g.nodes[mapping[expand]]['cluster'] != start_c:
                leaf_list.append(expand)
            else:
                for suc in g.predecessors(expand):
                    if suc not in explored:
                        if suc in frontier:
                            parent[suc].append(expand)
                        else:
                            parent[suc] = expand
                            frontier.append(suc)
                    else:
                        parent[suc].append(expand)
        return leaf_list, explored, parent
        #leaf_list is a list of tasks. explored is a list of tasks. parent is a dictionary
        #in the format of {task:task}
        #parent is in terms of BFS tree with root start_t
        # we need leaf to do duplication bfs. We need explored and parent to
        #identify the processors and edges to be duplicated



    def _dup_bfs_cluster(self,mapping,leaf_list,parent,end_c,bad_link, spdup, component_thp ):
        #this function returns the BFS path from the parent clusters of the source
        #cluster to the sink cluster of the bad link
        #end_c is the cluster of the sink of the bad link
        #spdup is a ratio that only inter-cluster link with throughput >= spdup*sys_throughput
        #will be selected in the BFS tree
        #parent, leaf_list are returned by function _dup_bfs_parent_cluster().
        #step1: BFS rooted with parent1,parent2, to check reachable clusters
        #step2: For common clusters in these BFS trees, check the availability, compute
        # throughput, and find dup_path one by one
        #To check availability, do BFS rooted with the candidate cluster, with parent1,
        #parent2, sink cluster of bad link as the goal node respectively. Flag a cluster as false once it is
        #in a BFS tree.
        #step3:pick the best candidate clsuter in terms of throughput to duplicate tasks
        processor_list = [mapping[leaf] for leaf in leaf_list]
        cluster_list = [self.processor_g.nodes[proc]['cluster'] for proc in processor_list]
        cluster_list_uniq = np.unique(np.array(cluster_list))
        # **** first make processor_list and cluster_list np array, then
        # **** cluster_proc_dup = {c: processor_list[np.where(cluster_list==c)] for c in cluster_list_uniq}
        cluster_proc_dup = {}
        for i in cluster_list_uniq:
            cluster_proc_dup[i] = []
        for idx, v in enumerate(cluster_list):
            cluster_proc_dup[v].append(processor_list[idx])
        #cluster_proc_dup is a dictionary in the format {cluster_dup:[processors_dup]}
        explored = {}
        #keep track of the explored clusters in the BFS tree for every start_c
        #in the format of {start_c:[explored clusters]}
        find_path = True
        mapping_rev = {v:k for k,v in mapping.items()}
        # reverse of mapping in the format of {processor:task}

        for start_c in cluster_list_uniq:
            if find_path == False:
                break
            _inter_link = []
            for _proc in cluster_proc_dup[start_c]:
                for p in parent[mapping_rev[_proc]]:
                    _inter_link.append((mapping_rev[_proc],p))
            _data_parent = max([self.task_dag.edges[e]['data'] for e in _inter_link])
            #_data_parent is the max of the inter_cluster links' data from cluster start_c to the source
            _data = max([_data_parent,self.task_dag.edges[bad_link]['data']])
            #_data also considers the commu data on the bad link.
            #cluster of the bad link
            #NB:e in the format of ('A','B'). Check if it can be treated as an edge
            #when calling self.task_dag.edges[e]


            frontier = [start_c]
            explored[start_c] = []
            find_path = False
            while frontier:
                expand = frontier.pop(0)
                explored[start_c].append(expand)

                for suc in self.cluster_g.successors(expand):
                    if suc == end_c and ((self.cluster_g.edges[(expand,suc)]['bandwidth']/_data) >= spdup*component_thp[bad_link]):
                        find_path = True
                        #NB:we flag find_path when start_c can reach end_c. But the clusters searched in the BFS\
                        #after this may not be able to reach end_c, thus useless. So maybe we should terminate the\
                        #search once end_c is reached, if performace is unsatisfying
                    if (suc not in explored[start_c]) and (suc not in frontier)\
                        and (self.cluster_g.nodes[suc]['procs'] == self.cluster_g.nodes[suc]['avai_procs'])\
                        and ((self.cluster_g.edges[(expand,suc)]['bandwidth']/_data) >= spdup*component_thp[bad_link]):
                        frontier.append(suc)

            explored[start_c].remove(start_c)

        #no valid path from start_c to end_c
        if find_path == False:
             return None,None,None
             #TODO:check the indentation of the 'if' above. The bug now is when calling bfs_find_dup_path, start_c = candidate, because the start_c is in the common_cluster.
        explored_list = list(explored.values())
        # **** alternative way of doing this:
        # **** explored_arr = np.concatenate(explored_list)
        # **** _idx,_cnt = np.unique(explored_arr,return_counts=True)
        # **** common_list = _idx[np.where(_cnt==len(explored_list))]
        common_cluster = set(explored_list[0])
        for c in explored_list[1:]:
            common_cluster.intersection_update(c)
            #common_list is the list of common clusters
        if common_cluster == set():
            find_path = False
            return None,None,None
        # ----step2-------------------------------------------------------------
        path_dup = {}
        #path_dup is a dic of dic. The format is
        #\{candidate:{(parent1,candi):[path],(candi,bad_cluster):[path],'throughput':2.38}}
        for candidate in common_cluster:
            find_path = True
            avai_clusters = list(self.cluster_g.nodes)
            #TODO: modify the above line.this avai_clusters is only valid in the first iteration of duplication
            path_dup[candidate] = {'throughput':float('inf')}
            for parent_c in list(cluster_list_uniq):
                #---for debug---------------------------------------------------------------------
                if parent_c == candidate:
                    import pdb; pdb.set_trace()
                #----------------------------------------------------------------------------------
                _path,_thp = self.bfs_find_dup_path(parent_c,candidate, avai_clusters,_data_parent, bad_link)
                if _path == []:
                    #NB:cannot find a path caused by some flagged clsuters.
                    path_dup[candidate]['throughput'] = 0
                    find_path = False
                    break

                path_dup[candidate][(parent_c,candidate)] = _path
                if _thp < path_dup[candidate]['throughput']:
                    path_dup[candidate]['throughput'] = _thp
                for i in _path:
                    avai_clusters.remove(i)
                avai_clusters.append(candidate)
            if find_path == False:
                continue
            _path,_thp = self.bfs_find_dup_path(candidate,end_c,avai_clusters,_data,bad_link)
            if _path == []:
                #NB:cannot find a path caused by some flagged clsuters.
                path_dup[candidate]['throughput'] = 0
                find_path = False
                continue
            path_dup[candidate][(candidate,end_c)] = _path
            if _thp < path_dup[candidate]['throughput']:
                path_dup[candidate]['throughput'] = _thp


        #NB:check whether candidate considers the reachability from bad cluster.

        #-----------------------------------------------------------------------
        #------step3-----------------------------------------------------------
        best_candidate = max(path_dup.keys(), key = (lambda x:path_dup[x]['throughput']))
        best_dup = path_dup[best_candidate]
        #best_candidate is a cluster.
        #best_dup is a value in path_dup. It is a dic in the format of\
        #{(p1,candidate):[path]...(candidate,end_c):[path],'throughput':2.38}
        if best_dup['throughput'] == 0:
            return None,None,None
        else:
            return best_candidate, best_dup, cluster_proc_dup


    def bfs_find_dup_path(self,start_c,end_c,avai_clusters,data,bad_link):
        frontier = [start_c]
        explored = []
        parent = {}
        find_path = False
        while frontier:
            expand = frontier.pop(0)
            explored.append(expand)

            for suc in self.cluster_g.successors(expand):
                if suc == end_c and ((self.cluster_g.edges[(expand,suc)]['bandwidth']/data) >= spdup*component_thp[bad_link]):
                    parent[suc] = expand
                    find_path = True
                    _path = self.traceback(start_c,end_c,parent)
                    break
                if (suc not in explored) and (suc not in frontier)\
                    and (self.cluster_g.nodes[suc]['procs'] == self.cluster_g.nodes[suc]['avai_procs'])\
                    and (suc in avai_clusters)\
                    and ((self.cluster_g.edges[(expand,suc)]['bandwidth']/data) >= spdup*component_thp[bad_link]):
                    frontier.append(suc)
                    parent[suc] = expand
            if find_path:
                break
        if not find_path:
            _path = []
            _thp = 0
        else:
            bw_list = [self.cluster_g.edges[_path[i],_path[i+1]]['bandwidth'] for i in range(len(_path)-1)]
            _thp = (min(bw_list))/data
        return _path, _thp


    def traceback(self, start_c, end_c, parent):
        #called by bfs_find_dup_path(). It returns the path from start cluster
        # to the end cluster. parent is a dictionary defined in the calling func.
        #path is a list.
        path = []
        path.append(end_c)

        try:
            while parent[path[-1]] != start_c:
                path.append(parent[path[-1]])
            path.append(start_c)
            path.reverse()
            return path

        except Exception:
            return None


    def _dup_bw_bottleneck(self, mapping,k, m, component_thp, sys_throughput):
        #this function returns the inter-clsuter links to be duplicated,
        #more precisely, returns edges in the task graph.
        # k is the max ratio of inter-clsuter links we want to duplicate
        # m is the ratio. if inter_cluster link l's throughput is less than
        #m* sys_throughput, we want to duplicate it.
        inter_links = {}
        # e.g.{('A','B'):[0,2,0]} in format of {edge:quadratic}
        for e in self.task_dag.edges:
            if self.processor_g.nodes[(mapping[e[0]])]['cluster'] != \
                self.processor_g.nodes[(mapping[e[1]])]['cluster']:
                inter_links[e] = self.processor_g.edges[mapping[e[0]],mapping[e[1]]]['quadratic']
                #NB:e is not in processor_g.edges!
        inter_links_sort = sorted(inter_links.keys(), key = lambda x:(1.0/inter_links[x][1]))
        # inter-cluster links sorted by increasing bandwidth
        num_dup = int(len(inter_links_sort)*k)
        inter_links_dup = inter_links_sort[0:num_dup]

        for l in inter_links_dup:
            if component_thp[l] > m*sys_throughput:
                #NB:check type of l should be an edge in task graph??
                inter_links_dup.remove(l)
        return inter_links_dup


    def dup(self,sys_thp,component_thp,mapping,k,m,spdup):
        #see k and m's definition in function _dup_bw_bottleneck()
        #spdup defined in _dup_bfs_cluster
        #NB: in duplication, we assume there are enough processors in a cluster to assign duplicated tasks or act
        #\as relay nodes. Need to check the assumption with test cases.
        mapping_rev = {v:k for k,v in mapping.items()}
        bad_links = self._dup_bw_bottleneck(mapping,k, m, component_thp, sys_thp)
        # >>>>>>>> crap
        bad_links = [bad_links[0],bad_links[1]]
        for count, bad_link in enumerate(bad_links):
            if bad_link[0] == 'A':
                #TODO: generalize 'A' to entry task
                continue
            leaf_list, explored, parent = self.s_cluster(mapping, bad_link)
            end_p = mapping[bad_link[1]]
            end_c = self.processor_g.nodes[end_p]['cluster']
            best_candidate, best_dup, cluster_proc_dup = self._dup_bfs_cluster(mapping,leaf_list,parent,end_c,bad_link, spdup, component_thp)
            if best_dup is None:
                continue
            #------remove bad link-------------------------------------------------------------------------------------------------------
            data_bad_link = self.task_dag.edges[bad_link]['data']
            self.task_dag.remove_edge(*bad_link)
            #-----modify the duplicated tasks(the candidate cluster)----------------------------------------------------------------------------------------------------
            dup_tasks_temp = list(set(explored) - set(leaf_list))
            dup_tasks = ['dup'+str(count)+'_'+t for t in dup_tasks_temp]
            #dup_tasks is the tasks duplicated in this round of duplication. task name
            #'dup0_A' means the replica of task A in the 0th round of duplication
            self.task_dag.add_nodes_from(dup_tasks)
            for t in dup_tasks:
                for i in list(self.processor_g.nodes):
                    self.processor_g.nodes[i]['comp'][t] = self.processor_g.nodes[i]['comp'][(t.split('_'))[1]]
                    #modify the processor_g's node 'comp' for the new tasks
                mapping[t] = self.cluster_g.nodes[best_candidate]['avai_procs'].pop(0)
                #modify mapping and cluster_g

            for t in dup_tasks:
                if t.split('_')[1] in parent.keys():
                    for p in parent[t.split('_')[1]]:
                        _d = self.task_dag.edges[t.split('_')[1],p]
                        self.task_dag.add_edges_from([(t,p,{'data':_d})])
                    #modify the edge's 'data' in task_dag

            #-----modify the path from candidate cluster to end cluster of the bad link--------------------
            thp_dup = best_dup['throughput']
            del best_dup['throughput']
            for k in best_dup.keys():
                if k[0] == best_candidate:
                    candi_to_end = k
                    #candi_to_end is the key of best_dup. It is in the format of
                    #(candi, end_c)
            cte_p = best_dup[candi_to_end]
            #candidate to end cluster path
            num_relay_cte = len(cte_p)-2
            relay_cte = ['dup'+str(count)+'_'+str(candi_to_end)+'R'+str(i) for i in range(num_relay_cte)]
            # task name 'dup0_(candi,end_c)R0' means the 0th relay node from candi_c to end_c
            #in the 0th duplication
            self.task_dag.add_nodes_from(relay_cte)

            for idx,t in enumerate(relay_cte):
                for i in list(self.processor_g.nodes):
                    self.processor_g.nodes[i]['comp'][t] = 0
                    #modify the processor_g's node 'comp' for the relay nodes to 0
                mapping[t] = self.cluster_g.nodes[cte_p[idx+1]]['avai_procs'].pop(0)
                #modify mapping and cluster_g


            edge_list = ['dup'+str(count)+'_'+bad_link[0]]+relay_cte+[bad_link[1]]
            #TODO: check the line above. It may cause errors in later duplication iterations for name of bad_link[0]
            for idx in range(len(edge_list)-1):
                self.task_dag.add_edges_from([(edge_list[idx],edge_list[idx+1],{'data':data_bad_link})])
                #add edges among the relay nodes from candidate cluster to end_c
            del best_dup[candi_to_end]

            #----modify the path from parent clusters to candidate cluster---------------------------------------------------------------------
            for k,v in best_dup.items():
                num_relay_ptc = len(v)-2
                #number of relay clusters from parent cluster to candidate cluster
                procs_p = cluster_proc_dup[k[0]]
                # parent processors in the parent cluster
                task_p = [mapping_rev[p] for p in procs_p]
                # parent tasks in the parent cluster
                for p in task_p:
                    for i in parent[p]:
                        _data = self.task_dag.edges[p,i]['data']
                        for j in range(num_relay_ptc):
                            relay_ptc = [('dup'+str(count)+'_'+str((p,i))+'R'+str(j)) for j in range(num_relay_ptc)]
                        if num_relay_ptc > 0:
                            self.task_dag.add_nodes_from(relay_ptc)
                            #NB:for each inter-cluster link from parent cluster to dupicated cluster, we use one relay node to commu.
                            #NB:\we assume relay nodes is less than the number of processors in a cluster. Need to check!!
                            #add relay nodes from parent cluster to candidate cluster

                            for idx, t in enumerate(relay_ptc):
                                for pro in list(self.processor_g.nodes):
                                    self.processor_g.nodes[pro]['comp'][t] = 0
                                    #modify the processor_g's node 'comp' for the new tasks
                                mapping[t] = self.cluster_g.nodes[v[idx+1]]['avai_procs'].pop(0)
                                #modify mapping and cluster_g

                        if num_relay_ptc > 1:
                            for k in range(num_relay_ptc-1):
                                self.task_dag.add_edges_from([(relay_ptc[k],relay_ptc[k+1],{'data':_data})])
                                #add edges between relay nodes
                        if num_relay_ptc != 0:
                            self.task_dag.add_edges_from([(p,'dup'+str(count)+'_'+str((p,i))+'R0',{'data':_data})])
                            self.task_dag.add_edges_from([('dup'+str(count)+'_'+str((p,i))+'R'+str(num_relay_ptc-1),'dup'+str(count)+'_'+i,{'data':_data})])
                            #add edge from parent cluster to relay. Add edge from relay to the candidate clsuter
                        else:
                            self.task_dag.add_edges_from([(p,'dup'+str(count)+'_'+i,{'data':_data})])
            print('mapping after dup {}\n{}'.format(count, mapping))

            self.sup_exec_g = nx.DiGraph()
            self.sup_exec_g.add_edges_from(self.task_dag.edges())
            for taskid in list(self.sup_exec_g.nodes):
                self.sup_exec_g.nodes[taskid]["in_proc"] = [mapping[taskid]]
                self.sup_exec_g.nodes[taskid]["weight"] = [1]
                #TODO: mapping is not a class variable for now. It works for the 0th duplication, But
                #\may cauze errors in the next iterations.
            min_thp, bottleneck, component_thp = self.get_sys_throughput()
            print('throughput after dup{}\n{}\n bottleneck is {}'.format(count,min_thp,bottleneck))


    def split(self):
        pass

    def schedule(self):
        #self._cluster() # optional
        self.dup()
        self.split()


#----generate processor graph by profiling----
def gen_processor_g(processor_names,sys_profile):

    processor_g = nx.DiGraph()
    processor_g.add_nodes_from(processor_names)
    for i in processor_names:
        processor_g.nodes[i]['comp'] = {k:v[i] for k,v in sys_profile['comp'].items()}
    processor_g.add_edges_from([(k[0],k[1],{"quadratic" :v})for k,v in sys_profile['link quad'].items()])

    nx.draw(processor_g, with_labels=True)
    plt.savefig('processor.png')
    return processor_g


#----generate task graph by tgff and profiling----
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
    #print(task_dag.edges.data())
    nx.draw(task_dag, with_labels=True)
    plt.savefig('task.png')

    return task_dag


def gen_exec_g(task_dag, processor_names):
    """
    helper function: for now just generate a dummy mapping of tasks onto processors,
    just to test the functionality of the scheduler class (e.g., throughput calculation, ...)

    OUTPUT:
        execution graph         nx graph of super-processors (same topology as task graph)
                                node attr: {'proc': [0,1,3], 'lambda': [w0,w1,w3]}
    """
    exec_g = nx.DiGraph()
    exec_g.add_edges_from(task_dag.edges())

    return exec_g

def parse_args():
    parser = argparse.ArgumentParser(description = "scheduling algorithm.")
    parser.add_argument("--path", type=str, default = PATH,
        help = "the path of the tgff file containing profiling information.")
    return parser.parse_args()

def get_initial_mapping(task_dag,processor_g,cluster_g):
    """
    input: task_dag and processor_g
    output:initial_mapping dictionary. initial_mapping = {'task1':1;'task2':2,'task3':6}
    ***the processor must be of type int
    """
    #initial_mapping = {'A': 0, 'B':1, 'C':2, 'D':3, 'E':4}
    #return initial_mapping


    avg_comp = dict()
    num_proc = len(list(processor_g.nodes()))
    for t in list(task_dag.nodes()):
        avg_comp[t] = 0
        for n in list(processor_g.nodes()):
            avg_comp[t] = avg_comp[t]+ processor_g.node[n]['comp'][t]
        avg_comp[t] = avg_comp[t]/num_proc
    #the average compution time of task t over all processors

    avg_com = dict()
    num_link = len(list(processor_g.edges()))
    for e in list(task_dag.edges()):
        _data = task_dag.edge[e]['data']
        avg_com[e] = 0
        for l in list(processor_g.edges()):
            _quad  = processor_g.edge[l]['quadratic']
            avg_com[e] = avg_com[e] + (_quad[0]*(_data**2)+_quad[1]*_data+_quad[2])
        avg_com[e] = avg_com[e]/num_link
    #the average commmunication time of e over all links

    _topo_t = list(nx.topological_sort(task_dag))
    uprank = dict()
    for t in _topo_t:
        uprank[t] = 0
    #NOTE:assume only one input task
    for t in _topo_t[1:]:
        _temp = max([(uprank[pre]+avg_com[(pre,t)]) for pre in task_dag.predecessors(t)])
        uprank[t] = avg_comp[t]+_temp
    dec_t = sorted(uprank.keys(), key = lambda x:-uprank[x])
    #dec_t is tasks in decreasing order of uprank
    avai_proc = list(processor_g.nodes())
    mapping = dict()
    for t in dec_t:
        sucs = list(task_dag.successors(t))
        data = [task_dag.edge[(t,_suc)]['data'] for _suc in sucs]
        #NOTE:check this
        min_occu = float('inf')
        for p in avai_proc:
        #NOTE:need to define avai_proc
            exit_flag = False
            #exit_flag = true if (p,mapping[_suc]) is not an edge in processor_g
            _comp = processor_g.node[p]['comp'][t]
            _max_comm = 0
            if len(sucs)!=0:
                for idx, _suc in enumerate(sucs):
                    _data = data[idx]
                    if processor_g.has_edge(p,mapping[_suc]) == False:
                        exit_flag = True
                        break

                    _quadra = processor_g.edges[p,mapping[_suc]]['quadratic']
                    _comm = _quadra[0]*(_data**2)+_quadra[1]*_data+_quadra[2]
                    if _comm > _max_comm:
                        _max_comm = _comm
                if exit_flag == True:
                    continue
            _occu = max(_comp,_max_comm)
            if _occu < min_occu:
                min_occu = _occu
                mapping[t] = p
        avai_proc.remove(mapping[t])
    return mapping
    #NOTE:maybe should record the resource occupation time to avoid redundant calculation of throughput later.








if __name__ == "__main__":
    args = parse_args()
    YAML = './proc_config.yml'
    """
    generating task dag and processor graph from system profile(tgff input)
    task_names, processor_names, sys_profile = utils.parse_prof_tgff(args.path)
    processor_g = gen_processor_g(processor_names,sys_profile)
    task_dag  = gen_task_dag(task_names,sys_profile)
    """
    ###---generate only task graph from tgff(sys profile)-----------------------

    task_names, processor_names, sys_profile = utils.parse_prof_tgff(args.path)
    task_dag  = gen_task_dag(task_names,sys_profile)
    cluster_g, processor_g, proc_generator = get_proc_graph_spec(YAML,task_names)
    scheduler = Scheduler(task_dag,processor_g,cluster_g)
    # manually add mapping from task nodes to processor nodes
    initial_mapping = get_initial_mapping(task_dag,processor_g,cluster_g)
    print("initial mapping{}".format(initial_mapping))
    scheduler._update_cluster_g(initial_mapping)

    for taskid,procid in initial_mapping.items():
        scheduler._add_mapping(procid,taskid,[1])

    proc_generator.visualize(sup_exec_g=scheduler.sup_exec_g,viz_outf='init.png')

    sys_thp, bottleneck, component_thp = scheduler.get_sys_throughput()
    print("initially, sys_throughput{}\n bottleneck{} \n component_thp{} \n".format(\
    sys_thp, bottleneck, component_thp))

    k = 3
    m = 1.5
    spdup = 1.03
    scheduler.dup(sys_thp,component_thp,initial_mapping,k,m,spdup)
    #NB:need to initialize and adjust k, m, spdup.
    proc_generator.visualize(sup_exec_g=scheduler.sup_exec_g,viz_outf='dup1.png')
    print('test above')
