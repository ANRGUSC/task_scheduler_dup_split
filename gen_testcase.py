import numpy as np
import argparse
import networkx as nx

import matplotlib.pyplot as plt

EPSILON = 1e-1

np.set_printoptions(precision=2)

class gen_proc_graph:
    """
    Assumption for processor graph:
    * Processors (Pi) are clustered
        * P[i][j] denote the jth processor in ith cluster
    * Link between processors
        * Use BW(P[i][j],P[a][b]) to denote the link bandwidth between P[i][j] and P[a][b]
        * BW(P[i][a],P[i][b]): large number (call it B_p)
        * BW(P[i][a],P[j][b]): small number (call it B_c)
        * BW(P[i][a],P[j][b]) = BW(P[i][n],P[j][m])
    Methodology for generating synthetic processor graph:
    * Generate cluster interconnection graph (with C clusters)
        * Goal: to fill the adj matrix $\in \mathbb{R}^{C\times C}$, representing
          the link bandwidth between clusters
        * Assume the variable B_c follows a Gaussian (?) distribution
    * Generate processor interconnection graph (with M_i processors in ith cluster)
        * Goal: to expand the cluster adj mat into a larger mat $\in \mathbb{R}^{CM\times CM}$
        * Assume the cluster size (M_i) follows a Gaussian (?) distribution
        * We already know all the inter-cluster BW, we want to generate the
          intra-cluster BW. For example, assume we have 3 clusters, with 3 processors
          in cluster 1, 2 processors in cluster 2 and 3 processors in cluster 3.

          o o o x x x x x
          o o o x x x x x
          o o o x x x x x
          x x x o o x x x
          x x x o o x x x
          x x x x x o o o
          x x x x x o o o
          x x x x x o o o

          where 'x's denote the entries we already know (based on cluster adj mat),
          and 'o's denote the entries we want to fill.
        * Assume the variable B_p follows a Gaussian (?) distribution
        * Processing power [to debate]:
            * Assume homogeneous processing power. i.e., computation time proportional
              to the workload attribute in task graph, independent of processor ID
    """
    def __init__(self, num_cluster, B_sigma_p, B_sigma_c, B_mu_p, B_mu_c, M_sigma, M_mu, C_sigma, C_mu):
        """
        num_cluster: number of clusters
        B_sigma_p: sigma for intra-cluster (inter-processor) bandwidth
        B_sigma_c: sigma for inter-cluster bandwidth
        M_sigma:   sigma for average cluster size (number of processors in a cluster)
        C_sigma:   sigma for computation power
        Similar for mu
        """
        # ---- user config ----
        self.num_cluster = num_cluster
        self.B_sigma_p = B_sigma_p
        self.B_sigma_c = B_sigma_c
        self.B_mu_p = B_mu_p
        self.B_mu_c = B_mu_c
        self.M_sigma = M_sigma
        self.M_mu = M_mu
        self.C_sigma = C_sigma
        self.C_mu = C_mu
        # ---- constants ----
        self.MIN_B_C = max(B_mu_c-2*B_sigma_c,0.1) #EPSILON 
        self.MAX_B_C = max(B_mu_c+2*B_sigma_c,0.1) #B_mu_c+B_sigma_c     # float('inf')
        self.MIN_B_P = max(B_mu_p-2*B_sigma_p,0.1) #EPSILON 
        self.MAX_B_P = max(B_mu_p+2*B_sigma_p,0.1) #B_mu_p+B_sigma_p     # float('inf')
        self.MIN_M   = int(max(M_mu-2*M_sigma,1)) #EPSILON 
        self.MAX_M   = int(max(M_mu+4*M_sigma,1)) #M_mu+M_sigma
        self.MIN_C   = max(C_mu-2*C_sigma,0.1) #EPSILON 
        self.MAX_C   = max(C_mu+2*C_sigma,0.1)
        #for normal distribution, 95% data are within the range (mu-2*sigma,mu+2*sigma)
        # ---- internal data struc ----
        self.cluster_adj = None
        self.proc_adj = None
        self.num_processor = None
        self.comp_rate = None
        self.proc_graph = None
        self.cluster_size = None
        self.cluster_graph = None

    def _gen_cluster_adj(self):
        _cluster_adj = np.random.normal(self.B_mu_c,self.B_sigma_c,\
                        self.num_cluster**2).reshape(self.num_cluster,self.num_cluster)
        _cluster_adj = np.clip(_cluster_adj,self.MIN_B_C,self.MAX_B_C)
        # NB: the diagnal should in fact be 0, but we don't care, since later on
        # we won't use the diagnal values anyways
        _cluster_size = np.random.normal(self.M_mu,self.M_sigma,self.num_cluster)
        _cluster_size = np.rint(_cluster_size).astype(int)
        _cluster_size = np.clip(_cluster_size,self.MIN_M,self.MAX_M)

        _cluster_graph = nx.DiGraph()
        _cluster_graph.add_nodes_from(list(range(self.num_cluster)))
        _end_id = np.cumsum(_cluster_size).astype(np.int)
        _start_id = np.concatenate(([0],_end_id))
        for i in range(self.num_cluster):
            _cluster_graph.nodes[i]['procs'] = [_start_id[i] + j for j in list(range(_cluster_size[i]))]
            _cluster_graph.nodes[i]['avai_procs'] = _cluster_graph.nodes[i]['procs']
        for m in range(self.num_cluster):
            for n in range(self.num_cluster):
                _cluster_graph.add_edges_from([(m,n,{'bandwidth':_cluster_adj[m][n]})])
        self.cluster_size = _cluster_size
        self.cluster_adj = _cluster_adj
        self.cluster_graph = _cluster_graph



    def _gen_proc_adj(self):
        num_processor = int(self.cluster_size.sum())
        _proc_adj = np.zeros((num_processor,num_processor))
        _end_id = np.cumsum(self.cluster_size).astype(np.int)
        _start_id = np.concatenate(([0],_end_id))
        for rid,_ in enumerate(_start_id[:-1]):
            for cid,__ in enumerate(_start_id[:-1]):
                _proc_adj[_start_id[rid]:_start_id[rid+1],_start_id[cid]:_start_id[cid+1]] = \
                self.cluster_adj[rid][cid]
        for diag_id,_ in enumerate(_start_id[:-1]):
            _link_block = np.random.normal(self.B_mu_p,self.B_sigma_p,self.cluster_size[diag_id]**2)
            _link_block = _link_block.reshape(self.cluster_size[diag_id],self.cluster_size[diag_id])
            _proc_adj[_start_id[diag_id]:_start_id[diag_id+1],_start_id[diag_id]:_start_id[diag_id+1]] = \
                        np.clip(_link_block,self.MIN_B_P,self.MAX_B_P)

        self.proc_adj = _proc_adj
        self.num_processor = num_processor

    def _gen_comp_rate(self):
        _comp_rate = np.random.normal(self.C_mu,self.C_sigma,self.num_processor)
        self.comp_rate = np.clip(_comp_rate, self.MIN_C,self.MAX_C)

    def _gen_proc_graph(self):
        _proc_graph = nx.DiGraph()
        _proc_graph.add_nodes_from(list(range(self.num_processor)))
        for i in range(self.num_processor):
            for j in range(self.num_processor):
                if i!=j:
                    _proc_graph.add_edges_from([(i,j,{'quadratic':[0,(1.0/self.proc_adj[i][j]),0]})])

        for i in range(self.num_processor):
            _proc_graph.nodes[i]['comp'] = self.comp_rate[i]
            #NB:need to modify the data structure of self.comp_rate to {'A':3,'B':5}

        for m in range(len(list(self.cluster_graph.nodes))):
            for n in self.cluster_graph.nodes[m]['procs']:
                _proc_graph.nodes[n]['cluster'] = m

        self.proc_graph = _proc_graph




def parse_args():
    parser = argparse.ArgumentParser(description = "generate processor graphs")
    parser.add_argument("--num_cluster", type=int, help="number of clusters")
    parser.add_argument("--B_mu_c", type=float, help = "mu for intercluster bw")
    parser.add_argument("--B_sigma_c", type=float, help = "sigma for intercluster bw")
    parser.add_argument("--B_mu_p", type=float, help = "mu for intracluster bw")
    parser.add_argument("--B_sigma_p", type=float, help = "sigma for intracluster bw")
    parser.add_argument("--M_mu", type=float, help = "mu for num processors in clusters")
    parser.add_argument("--M_sigma", type=float, help = "sigma for num processors in clusters")
    parser.add_argument("--C_mu", type=float, help = "mu for processing power")
    parser.add_argument("--C_sigma", type=float, help="sigma for processing power")
    return parser.parse_args()

def gen_testcase(num_cluster,B_sigma_p,B_sigma_c,\
        B_mu_p,B_mu_c,M_sigma,M_mu,C_sigma,C_mu):

    generator = gen_proc_graph(num_cluster,B_sigma_p,B_sigma_c,\
            B_mu_p,B_mu_c,M_sigma,M_mu,C_sigma,C_mu)
    generator._gen_cluster_adj()
    generator._gen_proc_adj()
    generator._gen_comp_rate()
    generator._gen_proc_graph()

    return generator.cluster_graph, generator.proc_graph







"""
if __name__ == "__main__":
    args = parse_args()
    generator = gen_proc_graph(args.num_cluster,args.B_sigma_p,args.B_sigma_c,\
            args.B_mu_p,args.B_mu_c,args.M_sigma,args.M_mu,args.C_sigma,args.C_mu)
    generator._gen_cluster_adj()
    generator._gen_proc_adj()
    generator._gen_comp_rate()
    generator._gen_proc_graph()
    import pdb; pdb.set_trace()
    print(generator.proc_adj)
    fig,ax = plt.subplots()
    im = ax.imshow(generator.proc_adj)
    plt.show()
"""
