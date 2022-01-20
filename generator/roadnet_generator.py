from hashlib import new
import os
import osmnx as ox
import numpy as np
import collections
import networkx as nx
import time
'''For graph simplification, please refer to https://osmnx.readthedocs.io/en/stable/osmnx.html#module-osmnx.simplification'''

speed_profile = {
    'bus_guideway': 50/3.6,
    'bus_stop': 50/3.6,
    'corridor': 50/3.6,
    'living_street': 20/3.6,
    'motorway': 80/3.6,
    'motorway_link': 60/3.6,
    'primary': 60/3.6,
    'primary_link': 60/3.6,
    'residential': 30/3.6,
    'road': 40/3.6,
    'secondary': 50/3.6,
    'secondary_link': 50/3.6,
    'service': 30/3.6,
    'tertiary': 40/3.6,
    'tertiary_link': 40/3.6,
    'track': 20/3.6,
    'trunk': 60/3.6,
    'trunk_link': 60/3.6,
    'unclassified': 35/3.6
}
default_speed = 30/3.6 

# bbox = {'north': 31.2711, 'south': 31.2281, 'east': 121.4300, 'west': 121.3515}

class Road_query:
    
    def __init__(self, bbox):
        self.G0 = ox.graph_from_bbox(bbox['north'], bbox['south'], bbox['east'], bbox['west'], network_type="drive", simplify=True)
        print(ox.stats.basic_stats(self.G0))
        self.G = nx.Graph(self.G0)
        print("initial length of edges: {} and nodes: {}".format(len(self.G.edges()), len(self.G.nodes())))
        self.set_speeds()
    
    def get_roadGraph(self):
        self.set_speeds()
        return self.G0, self.combine_edge(self.G)

    def set_speeds(self):
        for e in self.G.edges():
            highway_class = self.G.edges[e]['highway']
        
            try:
                maxspeed = self.G.edges[e]['maxspeed']
                self.G.edges[e]['speed_limit'] = int(maxspeed)
            except:
                if type(highway_class) != str:
                    highway_class = highway_class[0]
                if highway_class in speed_profile.keys():
                    speed_limit = speed_profile[highway_class]
                    self.G.edges[e]['speed_limit'] = speed_limit
                else:
                    self.G.edges[e]['speed_limit'] = default_speed # use universal speed_limit for all unclassified edges    
   
    def combine_edge(self, G):
        # find nodes with 3+ degrees
        node_degree_3plus = [n for n, d in G.degree() if d >= 3]
    #     node_degree1 = [n for n, d in G.degree() if d == 1]
        
        new_edges = []
        for nd in node_degree_3plus:
            merged_edge = self._merge_edge(nd, G)
            new_edges += merged_edge    
        
        new_G = nx.Graph(new_edges)
        
        # make sure there is no node with degree of 2, see following Example 1
        _, degreeCount = self.describe_degree(new_G)
        
        if degreeCount[2] > 0:
            new_G = self.combine_edge(new_G)
        
        print("after merge mid-edges, number of edges-{}, number of nodes-{}".format(len(new_G.edges()), len(new_G.nodes())))

        return new_G

    def describe_degree(self, G):
        degree_sequence = sorted([d for n, d in G.degree()])  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())
        total_cnt = sum(cnt)
        
        degreePercent = dict()
        for key, val in degreeCount.items():
            degreePercent[key] = val / total_cnt
        
        return degreePercent, degreeCount

    def _merge_edge(self, node, G):
        merged_edges = []

        for cur_node in G.adj[node]:
            cur_node_degree = G.degree(cur_node)
            
            
            nodes = [node]
            length = 0
            speed_limit = G[node][cur_node]['speed_limit']
            
            nodes, length, speed_limit = self._directed_search(G, cur_node, cur_node_degree, nodes, length, speed_limit)
            
            
            merged_edge = (node, nodes[-1], {'length': length, 'speed_limit': speed_limit})
            merged_edges.append(merged_edge)
            
        return merged_edges

    def _directed_search(self, G, cur_node, cur_node_degree, nodes, length, speed_limit):
        '''
        :prev_node ---> cur_node ---> next_node
        '''
        # update node_list, length, speed_limit for cur_node
        prev_node = nodes[-1] # node_list = [..., prev_node]
        nodes.append(cur_node)

        length += G[prev_node][cur_node]['length']
        speed_limit = min(G[prev_node][cur_node]['speed_limit'], speed_limit)
        
        # update cur_node with next_node or return
        if cur_node_degree == 2: # it is neither an intersection nor an endpoint
            
            # get next node data
            adj_nodes = list(G.adj[cur_node]) # [prev_node, next_node]
            adj_nodes.remove(prev_node) # remove prev_node, which is included in adj of cur_node
            
            next_node = adj_nodes[0]
            next_node_degree = G.degree(next_node)

            return self._directed_search(G, next_node, next_node_degree, nodes, length, speed_limit)
        else: # it is an intersection or endpoint
            return nodes, length, speed_limit    

class Road_generator:
    '''
    :simplify and translate OSM graphml file into roadnet.txt
    '''
    def __init__(self, input_file, output_file, bbox=None):
        # file_path = os.path.join(country_folder, file_name)
        self.G0 = None
        if bbox:
            rq = Road_query(bbox)
            self.G0, G0 =  rq.get_roadGraph()
        else:
            G0 = nx.read_graphml(input_file)
        print("finish query")
        self.G = nx.Graph(G0) # road network graph with 

        print("initial graph with {} edges".format(len(G0.edges())))
        self.output = output_file
        
        print("updated graph with {} edges".format(len(self.G.edges())))

        if bbox == None:
            self.bbox = {'south': 360, 'north': -360,  'west': 360, 'east': -360}
            self.no_bbox = True
        else:
            self.bbox = bbox
            self.no_bbox = False
            
        
    # def describe_graph(self, mode='all'):
    #     '''
    #     :describe initial graph before processing it
    #     '''
    #     # for nodes, describe node degrees
    #     degree_sequence = sorted([d for _, d in self.G.degree()])  # degree sequence
    #     degree_count = collections.Counter(degree_sequence)
        
    #     length_cat = {'l15': 0, '15_30': 0, '30_50': 0, 'g50':0}
        
    #     if mode == 'node_degree':
    #         return degree_count
    #     # for edges, describe highway classes and length categories
    #     highways = []
        
    #     for e in self.G.edges():
    #         # step-1: cluster edges by highway class
    #         h = self.G.edges[e]['highway']
    #         if type(h) != str:
    #             highways.append(h[0])
    #         else:
    #             highways.append(h)
            
    #         # step-2: cluster edges by length
    #         length = float(self.G.edges[e]['length'])
    #         if length <= 15:
    #             length_cat['l15'] += 1
    #         elif length <= 30:
    #             length_cat['15_30'] += 1
    #         elif length <= 50:
    #             length_cat['30_50'] += 1
    #         else:
    #             length_cat['g50'] += 1
            
    #     highway_count = collections.Counter(highways)
        
    #     return (degree_count, highway_count, length_cat)    

    def set_speeds(self):
        for e in self.G.edges():
            highway_class = self.G.edges[e]['highway']
        
            try:
                maxspeed = self.G.edges[e]['maxspeed']
                self.G.edges[e]['speed_limit'] = int(maxspeed)
            except:
                if type(highway_class) != str:
                    highway_class = highway_class[0]
                if highway_class in speed_profile.keys():
                    speed_limit = speed_profile[highway_class]
                    self.G.edges[e]['speed_limit'] = speed_limit
                else:
                    self.G.edges[e]['speed_limit'] = default_speed # use universal speed_limit for all unclassified edges
    
    def simplify_graph(self):
        # step-1 remove middle edges and merge middle road edges
        pass
        # step-2 Todo: identify and consolidate complex intersections
        

    def generate_roadnet(self):
        if self.no_bbox:
            self.set_speeds()
        # self.simplify_graph()
        node_data = self._get_node_data(self.G)
        edge_data, node_approach = self._get_edge_data(self.G)
        signal_data = self._get_signal_data(node_approach)

        network_length, num_nodes, num_signals, num_edges = self.describe_roadnet()
        # write road network data into txt file
        self.write_roadnet(self.output, node_data, edge_data, signal_data, num_edges)

        return (self.bbox, network_length, num_nodes, num_signals, num_edges)

    
    def describe_roadnet(self):
        '''
        :describe output road network data
        '''
        # step-1. compute network length
        network_length = 0
        for e in self.G.edges():
            network_length += float(self.G[e[0]][e[1]]['length'])
        
        # step-2: get number of nodes, signalized nodes and edges
        num_nodes = len(self.G.nodes())
        num_edges = len(self.G.edges())
        num_signals = len(self.signal_nodes)
        

        return network_length, num_nodes, num_signals, num_edges

    def write_roadnet(self, file_name, node_data, edge_data, signal_data, num_edges):
        '''
        :write the road network data'''
        fw = open(file_name, "w")
        fw.truncate()
        # part 1.0: node_num
        num_nodes = len(node_data)
        fw.write(str(num_nodes) + "\n")

        #part 1.1: node data
        for nd in node_data:
            for element in nd:
                fw.write(str(element) + " ")   
            fw.write("\n")
        
        #part 2.0 edge num
        fw.write(str(num_edges) + "\n")
        #part 2.1 edge data
        for ed in edge_data:
            for e in ed:
                fw.write(str(e) + " ")
            fw.write("\n")

        #part 3.0 signal num
        num_signals = len(signal_data)
        fw.write(str(num_signals) + "\n")
        for sig in signal_data:
            for s in sig:
                fw.write(str(s) + " ")
            fw.write("\n")
        fw.close()

    # get node, signal, edge data
    def _get_node_data(self, G):
        '''
        :parse node data and generate node data to write
        '''
        node_degrees = G.degree()

        # print(len(node_degrees))
        self.signal_nodes = []
        node_data = []
        # parsing nodes
        if self.no_bbox:
            for key, val in node_degrees:
                    # update min and max longitude and latitude for bbox
                long = float(G.nodes[key]['x'])
                lat = float(G.nodes[key]['y'])
                self._update_bbox(long, lat)

                if val >= 3 and val <= 4:
                    temp = (lat, long, key, 1)
                    node_data.append(temp)
                    self.signal_nodes.append(key)
                else:
                    temp = (lat, long, key, 0)
                    node_data.append(temp)
     
            
        else:
            for key, val in node_degrees:
                
                long = float(self.G0.nodes[key]['x'])
                lat = float(self.G0.nodes[key]['y'])

                if val >= 3 and val <= 4:
                    temp = (lat, long, key, 1)
                    node_data.append(temp)
                    self.signal_nodes.append(key)
                else:
                    temp = (lat, long, key, 0)
                    node_data.append(temp)      

        
        return node_data

    def _update_bbox(self, long, lat):
        '''
        : update min and max longitude and latitude for bbox
        '''
        if lat < self.bbox['south']:
            self.bbox['south'] = lat * 0.999
        
        if lat > self.bbox['north']:
            self.bbox['north'] = lat * 1.001

        if long < self.bbox['west']:
            self.bbox['west'] = long * 0.999
        
        if long > self.bbox['east']:
            self.bbox['east'] = long * 1.001

    def _get_edge_data(self, G):
        
        edge_temp = [(e[0], e[1], G[e[0]][e[1]]['length'], G[e[0]][e[1]]['speed_limit'], 3, 3) for e in G.edges()]
        
        edge_data = []
        node_approach = dict()
        count = 0
        for i in range(3 * len(edge_temp)):
            if i % 3 == 0:
                count += 2
                idx = int(i / 3)
                temp = edge_temp[idx] + (count-1, count)
            
                node0 = edge_temp[idx][0]
                node1 = edge_temp[idx][1]
            
                if node0 not in node_approach.keys():
                    node_approach[node0] = [count - 1]
                else:
                    node_approach[node0].append(count-1)
            
                if node1 not in node_approach.keys():
                    node_approach[node1] = [count]
                else:
                    node_approach[node1].append(count)
                
                edge_data.append(temp)
            else:
                lane_temp = (1, 0, 0, 0, 1, 0, 0, 0, 1)
                edge_data.append(lane_temp)
        
        return edge_data, node_approach

    def _get_signal_data(self, node_approach):
        
        signal_data = []
        for nd in node_approach.keys():
            if nd in self.signal_nodes:
                approaches = node_approach[nd]
                if len(approaches) < 4:   
                    temp = (nd, ) + tuple(approaches) + (-1, )
                else:
                    temp = (nd, ) + tuple(approaches)
                signal_data.append(temp)
            else:
                continue
                
        return signal_data

if __name__ == "__main__":
    # files = ['beijing-10687.graphml', 'changchun-11248.graphml', 'hangzhou-12386.graphml', 'shanghai-12400.graphml', 'nanchang-12010.graphml']
    # files = ['beijing-10687.graphml']
    # files = ['changchun-11248.graphml']
    # files = ['hangzhou-12386.graphml']
    # files = ['shanghai-12400.graphml']
    # files = ['nanchang-12010.graphml']

    # # file_path = os.path.join(country_folder, file_name)
    # # core_city, uc_id = filename.replace('.graphml', '').split('-')
    # in_folder = './input/'
    # out_folder = './output/'

    # t0 = time.time()

    # for file in files:
    #     print("start create data for {}".format(file.replace('.graphml', ' ')))
    #     input_file = os.path.join(in_folder, file)
    #     output_file = os.path.join(out_folder, file.replace('.graphml', '.txt'))
    #     rg = Road_generator(input_file, output_file)
    #     (bbox, network_length, num_nodes, num_signals, num_edges) = rg.generate_roadnet()
    #     (degree_count, highway_count, length_cat) = rg.describe_graph()
    
    # print("{} nodes, {} edges, {} signals, total_length: {}".format(num_nodes, num_edges, num_signals, network_length))
    # print("degree count: {}".format(degree_count))
    # print("length category: {}".format(length_cat))                                                                                                                   
    # t1 = time.time()
    # print("running time is {}".format(t1 - t0))
    rq = Road_query(bbox)
    roadG =  rq.get_roadGraph()
            
                