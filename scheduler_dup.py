import networkx as nx
import numpy as np
import networkx.drawing

from scheduler_general import *

import pdb

PRINT=False

class Scheduler_Dup(Scheduler_General):
    FAIL_CANDY = 'no candy found'
    FAIL_ROUTE = 'no route found'
    FAIL_CLUSIZE = 'too many links'
    FAIL_INDGR = 'input cluster cannot be dupped'

    def __init__(self, g_task, g_processor,
                k,m,spdup,num_reserve):
        super().__init__(g_task,g_processor,num_reserve=num_reserve)
        self.k = k
        #dup at most k number of bad links
        self.m = m
        #dup when the link throughput < m*sys_throughput
        self.spdup = spdup
        self._relay_count = 0
        self._code_failure = None


    def identify_bad_links(self):
        """
        step0: returns a dictionary of all bad links with low bw
        """
        bad_links = dict()
        for l in self.g_task2proc.edges:
            _thp = self.calc_link_throughput(l,None,None)[0]
            if _thp < self.m*self.sys_throughput:
                bad_links[l] = _thp
        #import pdb; pdb.set_trace()
        bad_links_sorted = [a for a,b in sorted(bad_links.items(), key=lambda kv:kv[1])]
        # sort by value. bad_links_sorted is a list of tuples
        num_bad_links = int(nx.number_of_edges(self.g_task2proc)*self.k)
        return bad_links_sorted[:num_bad_links]


    def _bfs(self,g,root,F_goal,F_terminate,F_neighbor):
        """
        returns
         * parents:     {node:parent}
         * function as an argument
        """
        frontier = [root]
        explored = []
        parent = {}
        is_goal = False
        while frontier:
            expand = frontier.pop(0)
            explored.append(expand)
            if F_goal(g,expand,explored,frontier):
                is_goal = True
            if F_terminate(g,expand,explored,frontier):
                break
            for suc in F_neighbor(g,expand,explored,frontier):
                frontier.append(suc)
                parent[suc] = expand
        return is_goal, parent, explored


    def _bfs_traceback_path(self,g,parent,start,end):
        """
        parent: dict mapping child to its parent
        """
        rev_path = [end]
        while parent[rev_path[-1]] != start:
            rev_path.append(parent[rev_path[-1]])
        rev_path.append(start)
        return list(reversed(rev_path))


    def dependency_traceback(self,bad_link):
        """
        step1: traceback the tasks/procs on which the source task of bad_link depends
        """
        root = self.g_task2proc.node[bad_link[0]]['proc_l'][0]  # root proc
        g = self.g_proc2task    #g_proc2task has the same topology with task_dag
        F_goal = lambda g,expand,explored,frontier: False
        F_terminate = lambda g,expand,explored,frontier: False
        F_neighbor = lambda g,expand,explored,frontier: [suc for suc in g.predecessors(expand) \
                        if suc not in explored and suc not in frontier and suc[0]==expand[0]]
        #only add nodes in the source cluster of bad_link as neighbors
        #a fuction as an argument of _bfs
        _, __, procs_to_dup = self._bfs(g,root,F_goal,F_terminate,F_neighbor)
        for p in procs_to_dup:
            assert p[0] == root[0]
        proc_links_to_dup = [(pp,p) for p in procs_to_dup for pp in g.predecessors(p)]
        #leaf nodes in parent clusters are covered in pp
        task_links_to_dup = [(g.node[l1]['task'],g.node[l2]['task']) for l1,l2 in proc_links_to_dup]
        inter_proc_links_to_dup = [(l1,l2) for l1,l2 in proc_links_to_dup if l1[0]!=l2[0]]
        intra_proc_links_to_dup = [(l1,l2) for l1,l2 in proc_links_to_dup if l1[0]==l2[0]]
        intra_task_links_to_dup = [(g.node[l1]['task'],g.node[l2]['task']) for l1,l2 in intra_proc_links_to_dup]
        parent_clusters = {l1[0]:0 for l1,l2 in inter_proc_links_to_dup}
        for l1,l2 in inter_proc_links_to_dup:
            parent_clusters[l1[0]] = max(g.edges[l1,l2]['data'],parent_clusters[l1[0]])
        #parent_clusters in the format of {cluster:max_data to source clsuter of bad_link}
        inter_proc_links_to_dup_dict = dict()
        inter_task_links_to_dup_dict = dict()
        for c in parent_clusters.keys():
            inter_proc_links_to_dup_dict[c] = [(l1,l2) for l1,l2 in inter_proc_links_to_dup if l1[0]==c]
            #inter_proc_links_to_dup_dict in the format of {cluster1:[inter_proc_links_to_dup starting from cluster1]}
            inter_task_links_to_dup_dict[c] = [(g.node[p1]['task'],g.node[p2]['task']) for p1,p2 in inter_proc_links_to_dup_dict[c]]
            #inter_task_links_to_dup_dict is similar to inter_proc_links_to_dup_dict. value is corresponding tasks
            if len(inter_proc_links_to_dup_dict[c]) > self.cluster_size:
                if PRINT:
                    print("inter_proc_links_to_dup from clsuter {} is larger than the cluster_size. Cannot dup!".format(c))
                return None
            #since in the relay clusters, one processor is used for one inter-clsuter link
        return parent_clusters, intra_task_links_to_dup, intra_proc_links_to_dup,\
                        inter_task_links_to_dup_dict, inter_proc_links_to_dup_dict


    def find_candidates(self,parent_clusters,bad_link,spdup_cur):

        """
        step2: find possible candidates for duplication
        """

        end_cluster = self.g_task2proc.node[bad_link[1]]['proc_l'][0][0]
        explored_all = []
        for clu,data in parent_clusters.items():
            root = clu
            g = self.g_cluster.copy()
            # ---- remove slow links of g ----
            _e_to_remove = []
            for l1,l2,d in g.edges(data=True):
                if d['bandwidth']/data < spdup_cur*self.calc_link_throughput(bad_link,None,None)[0]:
                    #adjust the parameter spdup
                    #NB:the bad_link throughput may be different from that before the 0th dup. If so,\
                    #we should not pick all the m bad links to dup in the beginning. Why not use component_thp here?
                    _e_to_remove.append((l1,l2))
            g.remove_edges_from(_e_to_remove)
            F_goal = lambda g,expand,explored,frontier: (expand==end_cluster) and (root!=end_cluster or len(explored)>1)
            F_terminate = lambda g,expand,explored,frontier: False
            #not terminate after finding goal
            F_neighbor = lambda g,expand,explored,frontier: \
                    [suc for suc in g.successors(expand) if suc not in explored \
                    and suc not in frontier and (self.is_empty_cluster(suc) or suc==end_cluster)] \
                    if expand!=end_cluster else []
                    #NB:the grammer above???
                    #not add successors of end_cluster as neighbors
            is_end_c_reached, _, explored = self._bfs(g,root,F_goal,F_terminate,F_neighbor)
            assert end_cluster in explored or not is_end_c_reached
            if (not is_end_c_reached) or (root not in explored):
                return None
            explored.remove(end_cluster)
            explored.remove(root)
            explored_all += explored
            #append explored clusters to explored_all iteratively
        explored_all = np.array(explored_all)
        _cand,_cnt = np.unique(explored_all,return_counts=True)
        return _cand[np.where(_cnt==len(parent_clusters))]
        #return clusters that can be reached by every parent cluster



    def find_routes(self,candy,parent_clusters,bad_link,spdup_cur,indgr0):
        """
        step3: find route and thp for a possible candidate
        """
        clu_taken = []
        ret = dict()
        _end_cluster = self.g_task2proc.node[bad_link[1]]['proc_l'][0][0]
        _data_bad_link = self.g_task2proc.edges[bad_link]['data']
        route_pairs = [{'start':k,'end':candy,'data':v} for k,v in parent_clusters.items()]
        route_pairs.append({'start':candy,'end':_end_cluster,'data':_data_bad_link})
        #route_pairs is a list of dictionaries. it is in the format of [{'start':clu_p1,'end':candy,'data':_data}]
        bad_link_throughput = self.calc_link_throughput(bad_link,None,None)[0]
        #NB:why not use component_thp here??
        ret = {'throughput': float('inf')}
        for ri in route_pairs:
            g = self.g_cluster.copy()
            #NB: if not copy, will g_cluster be changed??
            _e_to_remove = []
            for l1,l2,d in g.edges(data=True):
                if d['bandwidth']/ri['data'] < spdup_cur*bad_link_throughput:
                    #remove links with relativley small thp, instead of checking the condition while doing bfs
                    #adjust the parameter spdup
                    _e_to_remove.append((l1,l2))
            g.remove_edges_from(_e_to_remove)
            g.remove_nodes_from(clu_taken)
            F_goal = lambda g,expand,explored,frontier: expand == ri['end']
            F_terminate = lambda g,expand,explored,frontier: expand == ri['end']
            F_neighbor = lambda g,expand,explored,frontier: [suc for suc in g.successors(expand) \
                                if suc not in explored and suc not in frontier \
                                and (self.is_empty_cluster(suc) or suc==ri['end'])]
            found_candy,parent,new_clu_taken = self._bfs(g,ri['start'],F_goal,F_terminate,F_neighbor)
            if found_candy and len(parent) == 0:
                assert indgr0
            #found_candy could also mean there is valid path from candy to sink cluster of bad link
            if not found_candy:
                return None
            new_clu_taken.remove(ri['end'])
            clu_taken += new_clu_taken
            #clu_taken is the clusers taken by routes from clu_pi to candy...
            if ri['start'] == ri['end']:
                assert indgr0
                _path = []
            else:
                _path = self._bfs_traceback_path(g,parent,ri['start'],ri['end'])
            ret[(ri['start'],ri['end'])] = _path
            if ri['start'] == ri['end'] and indgr0:
                ret['throughput'] = float('inf')
            else:
                ret['throughput'] = min(self.calc_clu_path_throughput(_path,ri['data']),ret['throughput'])
            #NB:should I use component_thp???
            #ret in the format of {(clu_p1,candy):[path1],(candy,clu_end):[_path],'throughput':3.5}
        return ret


    def add_relay_path_to_candy(self,path,inter_proc_links,inter_task_links,dup_count):
        """
        path starts from an existing cluster, and ends at candy
        """
        num_links = len(inter_task_links)
        _F_convert = lambda lt1,lt2,lp1,lp2,i:[{'task_from':lt1[i],'task_to':lt2[i],\
                                                'proc_from':lp1[i],'proc_to':lp2[i]}]
        data_lookup = [self.g_task2proc.edges[l]['data'] for l in inter_task_links]
        F_comp_lookup = lambda t,p: self.g_processor.node[p]['comp'][t.split('#')[-1]]

        if len(path) == 2:
            #without relay nodes
            proc_from_l = [l1 for l1,_ in inter_proc_links]
            task_from_l = [t1 for t1,_ in inter_task_links]
            proc_to_l = [(path[-1],l2[1]) for _,l2 in inter_proc_links]
            task_to_l = ['D{}#{}'.format(dup_count,t2) for _,t2 in inter_task_links]
            for i,_l in enumerate(proc_from_l):
                F_data_lookup = lambda t1,t2,p1,p2: data_lookup[i]
                _links = _F_convert(task_from_l,task_to_l,proc_from_l,proc_to_l,i)
                self.add_links(_links,F_comp_lookup,F_data_lookup)
        else:
            #with relay nodes
            # parent to 1st relay
            proc_from_l = [l1 for l1,_ in inter_proc_links]
            proc_to_l = [(path[1],i) for i in range(num_links)]
            task_from_l = [t1 for t1,_ in inter_task_links]
            task_to_l = ['D{}-{}#RELAY'.format(dup_count,i) \
                    for i in range(self._relay_count,self._relay_count+num_links)]
            self._relay_count += num_links
            for i,_l in enumerate(proc_from_l):
                F_data_lookup = lambda t1,t2,p1,p2: data_lookup[i]
                _links = _F_convert(task_from_l,task_to_l,proc_from_l,proc_to_l,i)
                self.add_links(_links,F_comp_lookup,F_data_lookup)
            # add internal relays
            for r in range(len(path)-3):
                task_from_l = task_to_l
                task_to_l = ['D{}-{}#RELAY'.format(dup_count,i) \
                    for i in range(self._relay_count,self._relay_count+num_links)]
                self._relay_count += num_links
                proc_from_l = proc_to_l
                proc_to_l = [(path[r+2],i) for i in range(num_links)]
                for i,_l in enumerate(proc_from_l):
                    F_data_lookup = lambda t1,t2,p1,p2: data_lookup[i]
                    _links = _F_convert(task_from_l,task_to_l,proc_from_l,proc_to_l,i)
                    self.add_links(_links,F_comp_lookup,F_data_lookup)
            # add last relay to candy
            task_from_l = task_to_l
            proc_from_l = proc_to_l
            task_to_l = ['D{}#{}'.format(dup_count,t2) for _,t2 in inter_task_links]
            proc_to_l = [(path[-1],l2[1]) for _,l2 in inter_proc_links]
            for i,_l in enumerate(proc_from_l):
                F_data_lookup = lambda t1,t2,p1,p2: data_lookup[i]
                _links = _F_convert(task_from_l,task_to_l,proc_from_l,proc_to_l,i)
                self.add_links(_links,F_comp_lookup,F_data_lookup)


    def add_relay_path_from_candy(self,path,bad_link,dup_count,indgr0):
        bad_proc_link = (self.g_task2proc.node[bad_link[0]]['proc_l'][0],\
                         self.g_task2proc.node[bad_link[1]]['proc_l'][0])
        #bad link represented by a tuple of processors
        if indgr0:
            task_from = bad_link[0]
        else:
            task_from = 'D{}#{}'.format(dup_count,bad_link[0])
        #task_from is the duplicated source task of the bad link
        proc_from = (path[0],bad_proc_link[0][1])
        F_comp_lookup = lambda t,p: self.g_processor.node[p]['comp'][t.split('#')[-1]]
        F_data_lookup = lambda t1,t2,p1,p2: self.g_task2proc.edges[bad_link]['data']
        #data amount is equal to that on the bad_link
        for r in range(len(path)-2):
            task_to = 'D{}-{}#RELAY'.format(dup_count,self._relay_count)
            #e.g.D2-6#RELAY means the relay node is in the 2nd dup iteration, the sys has 6 relay nodes in total
            #TODO: check whether the naming method will cause problems
            self._relay_count += 1
            proc_to = (path[r+1],bad_proc_link[0][1])
            self.add_links([{'task_from':task_from,'task_to':task_to,'proc_from':proc_from,'proc_to':proc_to}],\
                            F_comp_lookup,F_data_lookup)
            task_from = task_to
            proc_from = proc_to
        task_to = bad_link[1]
        #task_to is the sink task of the bad_link
        proc_to = bad_proc_link[1]
        self.add_links([{'task_from':task_from,'task_to':task_to,'proc_from':proc_from,'proc_to':proc_to}],\
                        F_comp_lookup,F_data_lookup)


    def duplicate(self):
        """
        the overall flow of duplication
        """
        bad_links = self.identify_bad_links()
        _count_succeed_dup = 0.


        spdup_step = 0.05
        spdup_range = 0.5       # **** you may want to increase this
        #NB:need to adjust the 2 parameters above
        spdup_l = np.arange(self.spdup,self.spdup+spdup_range,spdup_step)
        spdup_l = np.flipud(spdup_l)

        for count,bad_link in enumerate(bad_links):

            indgr0 = False

            if self.g_task2proc.in_degree(bad_link[0]) == 0:
                #print("failed to dup bad link #{} with throughput {}, the input cluster cannnot be dupped".format(count,self.calc_link_throughput(bad_link)[0]))
                #if count == 0:
                #    self._code_failure = self.FAIL_INDGR
                #continue
                #NB:should be the in_degree of the cluster == 0 -- you can address this by adding a virtual source
                indgr0 = True
            bad_clu_from = self.g_task2proc.node[bad_link[0]]['proc_l'][0][0]
            bad_clu_to = self.g_task2proc.node[bad_link[1]]['proc_l'][0][0]
            # ---- get dependency tasks ----
            # parent_clusters: {cluster_id: max_data}
            _ret = self.dependency_traceback(bad_link)
            if _ret is None:
                if PRINT:
                    print("failed to dup bad link #{} with throughput {}, the cluster size is smaller than the number of inter-cluster links".format(count,self.calc_link_throughput(bad_link)[0]))
                if count == 0:
                    #only log the failure reason for the #0 link
                    self._code_failure = self.FAIL_CLUSIZE
                continue

            for try_i,spdup_cur in enumerate(spdup_l):
                parent_clusters, intra_task_links_to_dup, intra_proc_links_to_dup,\
                    inter_task_links_to_dup_dict, inter_proc_links_to_dup_dict = _ret
                #intra means in the source cluster of bad_link
                #inter means from parent cluster to the source cluster of bad_link


                if indgr0 == True:
                    candidates = [bad_clu_from]
                else:
                    candidates = self.find_candidates(parent_clusters,bad_link,spdup_cur)


                if candidates is None:
                    if count == 0 and try_i == len(spdup_l)-1:
                        #only log the failure reason for the #0 link
                        self._code_failure = self.FAIL_CANDY
                    if try_i == len(spdup_l)-1:
                        if PRINT:
                            print('failed to find candy, cannot dup bad link # {} with throughput {}'.format(count,self.calc_link_throughput(bad_link,None,None)[0]))
                    continue



                path_dup = dict()
                for candy in candidates:
                    paths_info = self.find_routes(candy,parent_clusters,bad_link,spdup_cur,indgr0)
                    #paths_info in the format of {(clu_p1,candy):[path1],(candy,clu_end):[_path],'throughput':3.5}
                    if paths_info is None:
                        continue
                    path_dup[candy] = paths_info
                    #path_dup in the format of {candy:paths_info}
                if len(path_dup) == 0:
                    if count == 0 and try_i == len(spdup_l)-1:
                        #only log the failure reason for the #0 link
                        self._code_failure = self.FAIL_ROUTE
                    if try_i == len(spdup_l)-1:
                        if PRINT:
                            print('failed to find paths for candy, cannot dup bad link # {} with throughput {}'.format(count,self.calc_link_throughput(bad_link,None,None)[0]))
                    continue
                sweet_candy = max(path_dup.keys(),key=lambda x:path_dup[x]['throughput'])
                sweet_candy_info = path_dup[sweet_candy]
                # sweet_candy_info in the format of {(clu_p1,candy):[path1],(candy,clu_end):[_path],'throughput':3.5}

                dup_throughput = sweet_candy_info.pop('throughput')
                #print('succeed in dup bad link # {}'.format(count))

                # ---- update data structures -----------------------------------------------------

                # --add cluster for dup tasks------
                if indgr0 == False:
                    intra_proc_links_to_dup = [((sweet_candy,l1[1]),(sweet_candy,l2[1])) \
                                                        for l1,l2 in intra_proc_links_to_dup]
                    #modify the processors' cluster ID to the sweet candy cluster's ID in intra_proc_links_to_dup
                    intra_task_links_to_dup = [('D{}#{}'.format(count,t1),'D{}#{}'.format(count,t2)) \
                                                        for t1,t2 in intra_task_links_to_dup]
                    #e.g.'D0#1' means 0th dup task 1
                    F_comp_lookup = lambda t,p: self.g_processor.node[p]['comp'][t.split('#')[-1]]
                    #F_comp_lookup returns the computation rate/time of task t on processor p
                    F_data_lookup = lambda t1,t2,p1,p2: self.g_task2proc\
                        .edges[t1.split('#')[-1],t2.split('#')[-1]]['data']
                    #F_data_lookup returns the communication data from task t1 to t2
                    _dup_intra_links = [{'proc_from':pp[0],'proc_to':pp[1],'task_from':tt[0],'task_to':tt[1]} \
                                    for pp,tt in zip(intra_proc_links_to_dup,intra_task_links_to_dup)]
                    #pp is an element in intra_proc_links_to_dup, tt is the corresponding element in intra_task_links_to_dup
                    self.add_links(_dup_intra_links,F_comp_lookup,F_data_lookup)

                #------add relays--------------------
                for k_dup_ends, v_dup_paths in sweet_candy_info.items():
                    if k_dup_ends[1] == sweet_candy:
                        if k_dup_ends[0] != sweet_candy:
                        # k_dup_ends[0] == k_dup_ends[1] == sweet_candy when indgr0 == True
                            try:
                                assert k_dup_ends[0] in parent_clusters.keys()
                            except Exception:
                                import pdb; pdb.set_trace()
                            start_cluster = k_dup_ends[0]
                            self.add_relay_path_to_candy(v_dup_paths,inter_proc_links_to_dup_dict[start_cluster],\
                                                    inter_task_links_to_dup_dict[start_cluster],count)
                    else:
                        assert k_dup_ends[1] == self.g_task2proc.node[bad_link[1]]['proc_l'][0][0]
                        assert k_dup_ends[0] == sweet_candy
                        self.add_relay_path_from_candy(v_dup_paths,bad_link,count,indgr0)
                #---remove bad link--------------------
                self.remove_links([{'task_from':bad_link[0],'task_to':bad_link[1]}])
                # ---- clean isolated tasks ----
                # NB: there is possibility that some old tasks are isolated after dup
                _count_succeed_dup += 1.
                break

        self.calc_overall_throughput(None,None)
        #the function above updates the self.sys_throughput
        if PRINT:
            print('done dup!')
        return _count_succeed_dup
