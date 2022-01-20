import os
# import pickle5 as pickle
import pickle
import time
from roadnet_generator import Road_generator
from traffic_generator import Traffic_generator
import argparse
import json
import networkx as nx
import pandas as pd

#todo, connect road_generator with traffic_generator directly, without write/read road txt file
def single_generate(flow_id, city, main_folder, cpus=4):
    # get.inputs

    flow_id = int(flow_id)
    start_flow = flow_id - 1
    end_flow = flow_id

    data_folder = main_folder + city + "/data/"
    
    road_json = data_folder + city + '.json'
    road_edge_pickle = data_folder + city + ".pickle"
    road_graphml_updated = data_folder + city + "_updated.graphml"
    
    road_graph = nx.read_graphml(road_graphml_updated)

    with open(road_edge_pickle, 'rb') as edge_pickle:
        edge_list = pickle.load(edge_pickle)
    print(edge_list[0])
    print("length of edges is {}".format(len(edge_list)))

    with open(road_json) as json_file:
        road_info = json.load(json_file)
        
    # generate traffic flow data for workers

    print("start generate traffic flows {} ... {}".format(start_flow, end_flow))
    tg = Traffic_generator(city, main_folder, road_info['bbox'], road_info['numveh'])
    tg.generate(edge_list, road_graph, runs=(start_flow, end_flow), cpus=cpus)    

# def master_generate(city, main_folder, bbox=None, traffic_duration=300, volume_param=0.1, cpus=16):
def master_generate(city, main_folder, bbox=None, traffic_duration=300, volume_param=0.4, cpus=16):
    '''
    : generator for master computer
    : tasks: 1) get roadnet.txt, 2) generate initial traffic flow (0_flow.txt)
    : inputs:
        city-ID
        

    '''
    # step-1 get roadnet.txt from graphml file
    # bbox = {'north': 31.2711, 'south': 31.2281, 'east': 121.4300, 'west': 121.3515}
    # bbox = {'north': 31.2639, 'south': 31.2323, 'east': 121.4151, 'west': 121.3629}
    # bbox = {'north': 31.2639, 'south': 31.2325, 'east': 121.4151, 'west': 121.3528}
    # bbox = {'north': 31.2644, 'south': 31.2262, 'east': 121.4093, 'west': 121.3562}
    graphml_folder = "./graphmls/"
    data_folder = main_folder + city + "/data/"
    cfg_folder = main_folder + city + "/cfg/"
    log_folder = main_folder + city + "/log/"
    
    create_folder(data_folder)
    create_folder(cfg_folder)
    create_folder(log_folder)
    
    data_folder = main_folder + city + "/data/"
    road_json = data_folder + city + '.json'

    road_graphml = graphml_folder + city + ".graphml"
    road_txt = data_folder + city + ".txt"

    # MIN_LAT-SOUTH, MAX_LAT-NORTH, MIN_LONG-WEST, MAX_LONG-EAST
    rg = Road_generator(road_graphml, road_txt, bbox)
    (bbox, network_length, num_nodes, num_signals, num_edges) = rg.generate_roadnet()
    
    print("{} nodes, {} edges, {} signals, total_length: {}".format(num_nodes, num_edges, num_signals, network_length))
    # print("degree count: {}".format(degree_count))
    # print("length category: {}".format(length_cat))
    
    # step - 2: first run to generate initial traffic flow, i.e., 0_flow.txt
    numveh = int(100000 * network_length / 1000000 * volume_param)                                                                      
    print("number of vehicle is {}".format(numveh))

    t0 = time.time()
    tg = Traffic_generator(city, main_folder, bbox, numveh, traffic_duration=traffic_duration)

    tg.initialize(cpus=cpus)
    t1 = time.time()
    print("running time is {}".format(t1-t0))
    print("{} edges after filter out with numveh < 5veh/min & length < 200m".format(num_edges))

    # step - 3: save data and paths to road_info.json
    data_dict = dict()
    data_dict['numveh'] = numveh
    data_dict['bbox'] = bbox
    data_dict['numedges'] = num_edges
    data_dict['running_time'] = t1 - t0

    with open(road_json, 'w') as f:
        json.dump(data_dict, f)

def create_folder(folder_name):
    isCreated = os.path.exists(folder_name)
    
    if not isCreated:
        os.makedirs(folder_name)
        print("{} is created.".format(folder_name))

if __name__ == "__main__":
    main_folder = "./"
    # mode - 1: get road and flow data from OSM bounding box (city.txt and 0_flow.txt)
    '''
    city = "manhattan"
    bbox = {'north': 40.7881, 'south': 40.7498, 'east': -73.9676, 'west':  -73.9821}
    '''
    
    city = "hangzhou"
    bbox = {'north': 30.2601,  'south': 30.2415, 'east': 120.1828, 'west':  120.1536}
    
    '''
    city = "shanghai"
    bbox = {'north': 31.2307,     'south': 31.2142,   'east':   121.4826, 'west':  121.4668}
    '''
    master_generate(city, main_folder, bbox, traffic_duration=3600, cpus=4)
    
    # 40.7150016, -74.0177709 Brooklyn battery city
    # 40.7351534, -73.9729007
    
    # by default, generate puto.txt and 0_flow.txt
    # cpus = 4 means python pooling cpu = 4, used to parallel generate traffic flow txt data
    
    # mode - 2: get road and flow data from graphml 
    # city = "changchun-11248"
    # bbox = None
    # master_generate(city, main_folder, bbox=None, traffic_duration=600)
    
    
  


