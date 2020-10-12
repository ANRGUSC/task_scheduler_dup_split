import numpy as np
import argparse
import networkx as nx
import yaml
import math

import matplotlib.pyplot as plt

#np.seterr(all='raise')

EPSILON = 1e-10
FONT = "/Library/Fonts/Arial.ttf"#'/usr/share/fonts/truetype/freefont/FreeMono.ttf'
#FONT = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"

"""
NOTE:

RIGHT NOW WE CANNOT ACCOMODATE THE PARAMETER MIN_DEG, CUZ

"""



class gen_cluster_graph:
    # *************************
    D0 =1
    #in meter
    KREF = -3
    PT = -32
    #in dBm
    # *************************
    def __init__(self,conf_clu,conf_env):
        """
        Generate <num_cluster> number of cluster in a field of <field_x> x <field_y> rectangle area.
        The area contains <num_obstacle> number of obstacles to communication.
        The location of the obstacles follow a uniform distribution within <field_x> x <field_y>.
        Each obstacle is a radius r circle, where r follows normal distribution defined by <r_obs_mu>, <r_obs_sigma>.
        No cluster can fall in the area of an obstacle.
        Clusters can only communicate with neighbor clusters. Cluster i is a neighbor of cluster j,
        if distance between i,j is smaller than <r_neigh>.
        """
        self.field_x = conf_clu['field_x']
        self.field_y = conf_clu['field_y']
        self.num_cluster = conf_clu['num_cluster']
        self.min_distance = conf_clu['min_distance']
        self.min_snr_db = conf_clu['min_snr_db']
        self.num_obstacle = conf_clu['num_obstacle']
        self.r_obs_mu = conf_clu['r_obs_mu']
        self.r_obs_sigma = conf_clu['r_obs_sigma']
        # ---- internal data structures ----
        self.pos_obs_array = np.zeros((self.num_obstacle,2))
        self.r_obs_array = np.zeros(self.num_obstacle)
        self.pos_cluster_array = np.zeros((self.num_cluster,2)) - self.min_distance
        #position of clusters in [(x,y)]
        self.N0 = conf_env['N0']
        self.ETA = conf_env['eta']
        self.psi = conf_env['psi']
        self.adj_cluster = np.zeros((self.num_cluster,self.num_cluster))
        self.r_neigh = self.get_r_neigh_max()

    # ********************************
    def get_snr(self,d):
        # calculate snr at distance d based on:
        # KREF, D0, self.N0
        Pr = self.PT+self.KREF-10*self.ETA*np.log10((d+EPSILON)/self.D0)
        if isinstance(Pr, float):
            offset = 0
        else:
            offset = np.random.normal(0,self.psi,len(Pr))
        Pr += offset
        #Pr = np.clip(Pr,EPSILON,None)


        return(Pr-self.N0)

    def get_bw(self,d):
        # calculate data rate at distance d based on:
        # self.get_snr(d)
        # value B in shannon equation can be anything, cuz this factor is offset by CCR also
        snr = self.get_snr(d)
        ret = np.log2(np.clip(1+snr,1,float('inf')))
        return ret

    def get_r_neigh_max(self):
        Pr_min = self.min_snr_db + self.N0
        r_neigh_min = 10**((self.PT+self.KREF-Pr_min)/(10*self.ETA))
        return(r_neigh_min*self.D0)

    # ********************************

    def gen_obstacles(self):
        _pos_x = np.random.uniform(0,self.field_x,self.num_obstacle)
        _pos_y = np.random.uniform(0,self.field_y,self.num_obstacle)
        self.pos_obs_array[:,0] = _pos_x
        self.pos_obs_array[:,1] = _pos_y
        self.r_obs_array = np.random.normal(self.r_obs_mu,self.r_obs_sigma,self.num_obstacle)

    def gen_cluster_pos(self):
        cur_num_cluster = 0
        for ci in range(1000*self.num_cluster):
            if cur_num_cluster == self.num_cluster:
                break
            _pos_x = np.random.uniform(0,self.field_x,1)
            _pos_y = np.random.uniform(0,self.field_y,1)
            _obs_x = self.pos_obs_array[:,0]
            _obs_y = self.pos_obs_array[:,1]
            _obs_dist = ((_obs_x-_pos_x)**2 + (_obs_y-_pos_y)**2)**0.5
            if np.any(_obs_dist<self.r_obs_array):
                continue
            _clu_x = self.pos_cluster_array[:,0]
            _clu_y = self.pos_cluster_array[:,1]
            _clu_dist = ((_clu_x-_pos_x)**2 + (_clu_y-_pos_y)**2)**0.5
            if np.any(_clu_dist<self.min_distance):
                continue
            self.pos_cluster_array[cur_num_cluster][0] = _pos_x
            self.pos_cluster_array[cur_num_cluster][1] = _pos_y
            cur_num_cluster += 1
        if len(np.where(self.pos_cluster_array<0)[0]) > 0:
            print('cannot generate cluster graph')
            return False
        else:
            return True

    def gen_cluster_adj(self):
        """
        TODO: if data rate is now a random variable, not dependent on distance, 
            then you need to control connectivity of clusters by, say, avg degree.
        """
        F_dist = lambda x1,x2,y1,y2: ((x1-x2)**2+(y1-y2)**2)**0.5
        clu_dist_mat = np.zeros((self.num_cluster,self.num_cluster))
        adj_clu_full = np.zeros((self.num_cluster,self.num_cluster))
        # ---- setup initial info ----
        for ci,pos_xy in enumerate(self.pos_cluster_array):
            _x,_y = pos_xy
            _clu_x = self.pos_cluster_array[:,0]
            _clu_y = self.pos_cluster_array[:,1]
            _clu_dist = F_dist(_clu_x,_x,_clu_y,_y)
            clu_dist_mat[ci] = _clu_dist
            adj_clu_full[ci] = self.get_bw(_clu_dist)
        # ---- neighbor based on absolute distance ----
        neigh_idx_dist = np.where(clu_dist_mat<= self.r_neigh)
        self.adj_cluster[neigh_idx_dist] = adj_clu_full[neigh_idx_dist]
        


class gen_proc_graph(gen_cluster_graph):
    def __init__(self,config_cluster,config_processor,conf_env):
        super().__init__(config_cluster,conf_env)
        self.clu_size = config_processor['clu_size']
        self.r_cluster = config_processor['r_cluster']
        self.comp_mu = config_processor['comp_mu']
        self.comp_sigma = config_processor['comp_sigma']
        # --------
        self.proc_pos = np.zeros((self.num_cluster,self.clu_size,2))
        self.proc_comp = np.zeros((self.num_cluster,self.clu_size))
        self.adj_proc = np.repeat(np.repeat(self.adj_cluster,self.clu_size,axis=0),self.clu_size,axis=1)

    def gen_proc_pos(self):
        F_dist = lambda x1,x2,y1,y2: ((x1-x2)**2+(y1-y2)**2)**0.

        for i, _pos_clu_xy in enumerate(self.pos_cluster_array):
            cur_num_proc = 0
            _pos_clu_x = _pos_clu_xy[0]
            _pos_clu_y = _pos_clu_xy[1]
            for j in range(100*self.clu_size):
                if cur_num_proc == self.clu_size:
                    break
                _pos_x = np.random.uniform(_pos_clu_x-self.r_cluster, _pos_clu_x+self.r_cluster,1)
                _pos_y = np.random.uniform(_pos_clu_y-self.r_cluster, _pos_clu_y+self.r_cluster,1)
                if F_dist(_pos_x,_pos_clu_x,_pos_y,_pos_clu_y) > self.r_cluster:
                    continue
                self.proc_pos[i][cur_num_proc][0] = _pos_x
                self.proc_pos[i][cur_num_proc][1] = _pos_y
                cur_num_proc += 1



    def gen_proc_adj(self):
        self.adj_proc = np.repeat(np.repeat(self.adj_cluster,self.clu_size,axis=0),self.clu_size,axis=1)
        F_dist = lambda x1,x2,y1,y2: ((x1-x2)**2+(y1-y2)**2)**0.5
        for i in range(self.num_cluster):
            _offset = i*self.clu_size
            for j in range(self.clu_size):
                for k in range(self.clu_size):
                    if k != j:

                        self.adj_proc[_offset+j][_offset+k] = self.get_bw(F_dist(self.proc_pos[i][j][0],self.proc_pos[i][k][0],self.proc_pos[i][j][1],self.proc_pos[i][k][1]))
                        #import pdb; pdb.set_trace()




    def gen_proc_comp(self):

        self.proc_comp = np.random.normal(self.comp_mu,self.comp_sigma,self.num_cluster*self.clu_size).reshape(self.num_cluster,self.clu_size)

    def to_nx_clu_graph(self):
        link_clu = np.vstack(np.where(self.adj_cluster>0)).T
        g_clu = nx.DiGraph()
        _edges = [(e[0],e[1],{'bandwidth':self.adj_cluster[e[0]][e[1]],'quadratic':(0,1/self.adj_cluster[e[0]][e[1]],0)}) for e in link_clu if e[0]!=e[1]]
        g_clu.add_edges_from(_edges)
        g_clu.add_nodes_from([i for i in range(self.num_cluster)])
        for i in range(self.num_cluster):
            try:
                g_clu.node[i]['procs'] = list(i*self.clu_size + np.arange(self.clu_size))
            except Exception:
                import pdb; pdb.set_trace()
            g_clu.node[i]['avai_procs'] = list(i*self.clu_size + np.arange(self.clu_size))
        return g_clu

    def to_nx_proc_graph(self,task_nodes):
        task_nodes = dict(task_nodes)
        link_proc = np.vstack(np.where(self.adj_proc>0)).T
        g_proc = nx.DiGraph()
        _edges = [((e[0]//self.clu_size,e[0]%self.clu_size),(e[1]//self.clu_size,e[1]%self.clu_size),\
                {'bandwidth':self.adj_proc[e[0]][e[1]],'quadratic':(0,1/self.adj_proc[e[0]][e[1]],0)}) for e in link_proc if e[0]!=e[1]]
        g_proc.add_edges_from(_edges)
        for i in range(self.num_cluster):
            for j in range(self.clu_size):
                g_proc.node[(i,j)]['comp'] = {t:d['comp']/self.proc_comp[i][j] for t,d in task_nodes.items()}
                g_proc.node[(i,j)]['comp']['RELAY'] = 0.
                g_proc.node[(i,j)]['cluster'] = i
                #NB: modify the data structure above
        return g_proc

    def _viz_link(self,draw,x_from,y_from,x_to,y_to,text,color,width,font,show_bw=False):
        _l_arrow = 20
        draw.line([x_from,y_from,x_to,y_to],fill=color,width=width)
        _alpha = np.arctan(abs(y_from-y_to)/abs(x_from-x_to))
        _dx_arrow = _l_arrow*np.cos(_alpha)
        _dy_arrow = _l_arrow*np.sin(_alpha)
        _dx_sgn = -1 if x_from<x_to else 1
        _dy_sgn = -1 if y_from<y_to else 1
        draw.line([x_to+_dx_sgn*_dx_arrow,y_to+_dy_sgn*_dy_arrow,x_to,y_to],fill=color,width=8)
        _x_mid = (x_from+x_to)/2.
        _y_mid = (y_from+y_to)/2.
        if show_bw:
            draw.text([_x_mid,_y_mid],"{:4.2f}".format(text),fill=color, font=font)

    def visualize(self,sup_exec_g=None,viz_outf='default.png',show_bw=False):
        import PIL.Image as I
        import PIL.ImageDraw as ID
        import PIL.ImageFont as IF
        font = IF.truetype(FONT,18) # "/Library/Fonts/Arial.ttf",24)
        scale_field = 10
        img = I.new("RGB", (scale_field*self.field_x,scale_field*self.field_y),"white")
        draw = ID.Draw(img)
        for oi,ri in enumerate(self.r_obs_array):
            _x,_y = scale_field*self.pos_obs_array[oi]
            ri *= scale_field
            draw.ellipse([_x-ri,_y-ri,_x+ri,_y+ri],fill="grey",outline="grey")
        #if sup_exec_g is None:
        _r_clu = 10
        #for c_xi,c_yi in self.pos_cluster_array:
        #    draw.ellipse([c_xi-_r_clu,c_yi-_r_clu,c_xi+_r_clu,c_yi+_r_clu],fill="black",outline="black")
        for c_from in range(self.num_cluster):
            for c_to in range(self.num_cluster):
                link = self.adj_cluster[c_from][c_to]
                if link == 0 or np.isinf(link):
                    continue
                _x_from = scale_field*self.pos_cluster_array[c_from][0]
                _y_from = scale_field*self.pos_cluster_array[c_from][1]
                _x_to = scale_field*self.pos_cluster_array[c_to][0]
                _y_to = scale_field*self.pos_cluster_array[c_to][1]
                if _x_from == _x_to and _y_from  == _y_to:
                    continue
                self._viz_link(draw,_x_from,_y_from,_x_to,_y_to,link,"#7ea8c8",2,font,show_bw=show_bw)






        if sup_exec_g is not None:
            #import pdb; pdb.set_trace()
            for supn in list(sup_exec_g.nodes(data=True)):
                _clu = supn[1]['proc_l'][0][0]
                _c_x,_c_y = self.pos_cluster_array[_clu]
                _c_x *= scale_field
                _c_y *= scale_field
                draw.ellipse([_c_x-_r_clu,_c_y-_r_clu,_c_x+_r_clu,_c_y+_r_clu],fill="#f6821f",outline="#f6821f")
                _neighs = list(sup_exec_g.neighbors(supn[0]))
                for n in _neighs:
                    _clu_to = sup_exec_g.node[n]['proc_l'][0][0]
                    _c_x_to,_c_y_to = scale_field*self.pos_cluster_array[_clu_to]
                    self._viz_link(draw,_c_x,_c_y,_c_x_to,_c_y_to,self.adj_cluster[_clu][_clu_to],"#f6821f",4,font,show_bw=True)

            for supn in list(sup_exec_g.nodes(data=True)):
                _clu = supn[1]['proc_l'][0][0]
                _c_x,_c_y = scale_field*self.pos_cluster_array[_clu]
                draw.text([_c_x,_c_y],"{}".format(supn[0]),fill="#006600",font=font)

        _off = 15
        for i,(c_xi,c_yi) in enumerate(self.pos_cluster_array):
            draw.text([scale_field*c_xi-_off,scale_field*c_yi-_off],str(i),fill="#000000",font=font)
        img.save(viz_outf,viz_outf.split(".")[-1],dpi=[500,500])






def parse_args():
    parser = argparse.ArgumentParser(description="generate cluster interconnection graph")
    parser.add_argument("--conf",type=str,required=True,help="yaml file specifying parameters of the generated cluster graph")
    #parser.add_argument("--viz_outf",type=str,required=False,default="test.png",help="output file name for the visualization plot")
    return parser.parse_args()



def get_proc_graph_spec(config_yml,task_nodes):
    with open(config_yml) as f_config:
        config = yaml.load(f_config)
    # ---- generate processors ----
    proc_generator = gen_proc_graph(config['cluster'],config['processor'],config['env'])
    proc_generator.gen_obstacles()
    if not proc_generator.gen_cluster_pos():
        return None
    proc_generator.gen_cluster_adj()
    proc_generator.gen_proc_pos()
    proc_generator.gen_proc_adj()
    proc_generator.gen_proc_comp()
    #proc_generator.visualize(viz_outf='proc.png')
    return proc_generator.to_nx_clu_graph(), proc_generator.to_nx_proc_graph(task_nodes), proc_generator




if __name__ == "__main__":
    args = parse_args()
    task_nodes = {'a':10,'b':20}
    g_clu,g_proc, proc_generator = get_proc_graph_spec(args.conf,task_nodes)