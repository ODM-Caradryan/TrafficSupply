#-*-coding:utf-8 -*-

import CBEngine
import gym
import agent.gym_cfg as gym_cfg
from agent.agent import FormulaAgent, FixedTimeAgent, MPAgent
from agent.mixAgent import MixAgent
import json
from pathlib import Path
import time
import logging
import networkx as nx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--setting',type=str,default='hangzhou')
args=parser.parse_args()

logging.basicConfig(level=logging.INFO, filename="./test.log" )
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
gym.logger.setLevel(gym.logger.ERROR)

simulator_cfg_file = './cfg/simulator.cfg'
gym_cfg_instance = gym_cfg.gym_cfg()

def read_config(cfg_file):
    configs = {}
    with open(cfg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if(len(line) == 3 and line[0][0] != '#'):
                configs[line[0]] = line[-1]
    return configs

def process_roadnet(roadnet_file):
    # intersections[key_id] = {
    #     'have_signal': bool,
    #     'end_roads': list of road_id. Roads that end at this intersection. The order is random.
    #     'start_roads': list of road_id. Roads that start at this intersection. The order is random.
    #     'lanes': list, contains the lane_id in. The order is explained in Docs.
    # }
    # roads[road_id] = {
    #     'start_inter':int. Start intersection_id.
    #     'end_inter':int. End intersection_id.
    #     'length': float. Road length.
    #     'speed_limit': float. Road speed limit.
    #     'num_lanes': int. Number of lanes in this road.
    #     'inverse_road':  Road_id of inverse_road.
    #     'lanes': dict. roads[road_id]['lanes'][lane_id] = list of 3 int value. Contains the Steerability of lanes.
    #               lane_id is road_id*100 + 0/1/2... For example, if road 9 have 3 lanes, then their id are 900, 901, 902
    # }
    # agents[agent_id] = list of length 8. contains the inroad0_id, inroad1_id, inroad2_id,inroad3_id, outroad0_id, outroad1_id, outroad2_id, outroad3_id

    intersections = {}
    roads = {}
    agents = {}

    agent_num = 0
    road_num = 0
    signal_num = 0
    with open(roadnet_file, 'r') as f:
        lines = f.readlines()
        cnt = 0
        pre_road = 0
        is_obverse = 0
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if ('' in line):
                line.remove('')
            if (len(line) == 1):
                if (cnt == 0):
                    agent_num = int(line[0])
                    cnt += 1
                elif (cnt == 1):
                    road_num = int(line[0]) * 2
                    cnt += 1
                elif (cnt == 2):
                    signal_num = int(line[0])
                    cnt += 1
            else:
                if (cnt == 1):
                    intersections[int(line[2])] = {
                        'have_signal': int(line[3]),
                        'end_roads': [],
                        'start_roads': [],
                        'lanes':[]
                    }
                elif (cnt == 2):
                    if (len(line) != 8):
                        road_id = pre_road[is_obverse]
                        roads[road_id]['lanes'] = {}
                        for i in range(roads[road_id]['num_lanes']):
                            roads[road_id]['lanes'][road_id * 100 + i] = list(map(int, line[i * 3:i * 3 + 3]))
                        is_obverse ^= 1
                    else:
                        roads[int(line[-2])] = {
                            'start_inter': int(line[0]),
                            'end_inter': int(line[1]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[4]),
                            'inverse_road': int(line[-1])
                        }
                        roads[int(line[-1])] = {
                            'start_inter': int(line[1]),
                            'end_inter': int(line[0]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[5]),
                            'inverse_road': int(line[-2])
                        }
                        intersections[int(line[0])]['end_roads'].append(int(line[-1]))
                        intersections[int(line[1])]['end_roads'].append(int(line[-2]))
                        intersections[int(line[0])]['start_roads'].append(int(line[-2]))
                        intersections[int(line[1])]['start_roads'].append(int(line[-1]))
                        pre_road = (int(line[-2]), int(line[-1]))
                else:
                    # 4 out-roads
                    signal_road_order = list(map(int, line[1:]))
                    now_agent = int(line[0])
                    in_roads = []
                    for road in signal_road_order:
                        if (road != -1):
                            in_roads.append(roads[road]['inverse_road'])
                        else:
                            in_roads.append(-1)
                    in_roads += signal_road_order
                    agents[now_agent] = in_roads
    for agent, agent_roads in agents.items():
        intersections[agent]['lanes'] = []
        for road in agent_roads:
            ## here we treat road -1 have 3 lanes
            if (road == -1):
                for i in range(3):
                    intersections[agent]['lanes'].append(-1)
            else:
                for lane in roads[road]['lanes'].keys():
                    intersections[agent]['lanes'].append(lane)

    return intersections, roads, agents

def get_graph(intersections, roads, agents):
    DG = nx.DiGraph()
    nodes = {}
    edges = []
    # for k,v in roads.items():
    #     for lane in v['lanes'].keys():
    #         nodes[lane] = {
    #             "demand":[],
    #             "supply":[],
    #             "target":[],
    #             "start_inter":v['start_inter'],
    #             "end_inter":v['end_inter'],
    #             'length':v['length'],
    #             'speed_limit':v['speed_limit'],
    #             "inverse_road":v['inverse_road']
    #         }
    cnt = 0
    for road,v in roads.items():
        if(v['length']<10.0):
            cnt +=1
    print("{} roads have length less than 10.0".format(cnt))
    for agent,agent_roads in agents.items():
        in_roads = agent_roads[:4]
        out_roads = agent_roads[4:]
        for idx, road in enumerate(in_roads):
            if(road == -1):
                continue
            for lane_id in range(3):
                lane = int(road*100+lane_id)
                v = roads[road]
                nodes[lane] = {
                    "demand": [],
                    "supply": [],
                    "target": [],
                    "start_inter": v['start_inter'],
                    "end_inter": v['end_inter'],
                    'length': v['length'],
                    'speed_limit': v['speed_limit'],
                    "inverse_road": v['inverse_road'],
                    "have_signal": True
                }
                tar_road = out_roads[(lane_id + idx + 1)%4]
                if(tar_road == -1):
                    continue
                for tar_lane_id in range(3):
                    tar_lane = int(tar_lane_id + tar_road*100)
                    edges.append((lane,tar_lane))

    for intersection, v in intersections.items():
        if(v['have_signal'] == True):
            continue
        for road in v['end_roads']:
            for lane_id in range(3):
                lane = int(road) * 100 + lane_id
                if(lane in nodes.keys()):
                    continue
                road_info = roads[road]

                nodes[lane] = {
                    "demand": [],
                    "supply": [],
                    "target": [],
                    "start_inter": road_info['start_inter'],
                    "end_inter": road_info['end_inter'],
                    'length': road_info['length'],
                    'speed_limit': road_info['speed_limit'],
                    "inverse_road": road_info['inverse_road'],
                    'have_signal':False
                }
                for start_road in v['start_roads']:
                    if(start_road == roads[road]['inverse_road'] ):
                        continue
                    for start_lane_id in range(3):
                        start_lane = start_road * 100 + start_lane_id

                        edges.append((lane,start_lane))

    for k, v in nodes.items():
        DG.add_node(k)
        for key, val in v.items():
            DG.nodes[k][key] = val
    DG.add_edges_from(edges)
    return DG

def main():
    env = gym.make(
        'CBEngine-v0',
        simulator_cfg_file=simulator_cfg_file,
        thread_num=1,
        gym_dict=gym_cfg_instance.cfg,
        metric_period = 36000
    )
    config={
        'time_interval' : 60
    }
    env.set_log(0)
    env.set_warning(0)
    env.set_ui(0)
    env.set_info(0)
    agent_id_list = []
    observations, infos = env.reset()
    
    # print(observations)
    
    
    
    for k in observations:
        agent_id_list.append(int(k.split('_')[0]))
    agent_id_list = list(set(agent_id_list))
    
    MP = MPAgent()
    FT = FixedTimeAgent(fixed_time=60)
    Formula = FormulaAgent()
    seed_dict = {
        'policy1': [0, 1, 2, 0],
        'policy2': [1, 2, 0, 1]
    }
    agent = MixAgent([FT, Formula, MP], seed_dict['policy1'])
    
    agent.load_agent_list(agent_id_list)
    simulator_configs = read_config(simulator_cfg_file)
    mx_step = simulator_configs['max_time_epoch']
    roadnet_path = Path(simulator_configs['road_file_addr'])
    intersections, roads, agents = process_roadnet(roadnet_path)
    agent.load_roadnet(intersections, roads, agents)
    done = False
    
    # print(agents)
    
    
    ##################################
    # graph
    DG = get_graph(intersections,roads,agents)

    ##################################
    # simulation
    step = 0
    log_path = Path(simulator_configs['report_log_addr'])
    sim_start = time.time()
    tot_v  = -1
    d_i = -1

    # calc ratio
    calc_outvehicles = {}
    lane2greesec = {}
    for k, v in DG.edges.items():
        calc_outvehicles[k] = 0
        DG.edges[k]['ratio'] = []
    for k in DG.nodes.keys():
        lane2greesec[k] = 0
        DG.nodes[k]['greensec'] = []
        DG.nodes[k]['innum']=[]
        DG.nodes[k]['outnum'] = []
        DG.nodes[k]['volume_split'] = []
        DG.nodes[k]['occupancy'] = []
        DG.nodes[k]['in_vehicle_hops'] = [[0,0,0,0,0,0,0,0]]
    cur_demand = {}
    for k, v in DG.nodes.items():
        cur_demand[k] = 0
    # calc volume and speed
    lane2volume = {}
    lane2speed = {}
    invehicles = {}
    outvehicles = {}
    minute_snapshot = {}
    invehicles_id = {}
    for k in DG.nodes.keys():
        lane2volume[k] = 0
        lane2speed[k] = [0,0]
        invehicles[k] = 0
        outvehicles[k] = 0
        invehicles_id[k] = []
    pre_vehicles = {}
    step = 0
    while not done:
        actions = {}
        # print(step)
        all_info = {
            'observations':observations,
            'info':infos,
            'cur_time': step,
            'eng': env.eng
        }
        # logger.info("step : {}, avg_tt : {}".format(step, env.eng.get_average_travel_time()))
        actions = agent.act(all_info)
        # print(actions)
        observations, rewards, dones, infos = env.step(actions)

        # collect data
        cur_vehicle_info = {}
        eng = env.eng
        vehicles = eng.get_vehicles()
        lane_vehicles = eng.get_lane_vehicles()
        for vehicle in vehicles:
            cur_vehicle_info[vehicle] = eng.get_vehicle_info(vehicle)

        cur_phases = {}
        for agent_id, phase in actions.items():
            cur_phases[int(agent_id)] = int(phase)

        for agent_id in agent_id_list:
            if(dones[agent_id]):
                done = True
        step += 1
        logging.info("{}/{}".format(step,mx_step))
        logging.info(lane_vehicles)
    '''
    '''

if __name__ == '__main__':
    main()

