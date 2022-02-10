#-*-coding:utf-8 -*-

import CBEngine
import gym
import agent.gym_cfg as gym_cfg
from agent.agent import FormulaAgent, FixedTimeAgent, MPAgent
import json
from pathlib import Path
import time
import logging
import networkx as nx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--setting',type=str,default='hangzhou')

args=parser.parse_args()


logging.basicConfig(level=logging.INFO)
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

def calc_inoutnum(time_interval,pre_vehicles,vehicles,step,invehicles,outvehicles,DG,minute_snapshot,invehicles_id):
    if(step>0):
        for vehicle, info in vehicles.items():
            if(vehicle not in pre_vehicles.keys()):
                continue
            if(info['route'] != pre_vehicles[vehicle]['route']):
                cur_lane = int(info['drivable'][0])
                pre_lane = int(pre_vehicles[vehicle]['drivable'][0])
                invehicles[cur_lane] += 1
                outvehicles[pre_lane] += 1

                ########## for FDTG case study
                if (not vehicle in minute_snapshot.keys()):
                    continue
                source_tm = int(minute_snapshot[vehicle]['drivable'][0])
                try:
                    distance = min(7, len(nx.shortest_path(DG, source=source_tm, target=cur_lane)) - 1)
                except Exception as e:
                    print(e)
                    continue
                # print(distance)
                DG.nodes[cur_lane]['in_vehicle_hops'][-1][distance] += 1

        if(step + 1) % time_interval == 0:
            for now_lane in DG.nodes.keys():
                if(now_lane in invehicles.keys()):
                    DG.nodes[now_lane]['innum'].append(invehicles[now_lane])
                if(now_lane in outvehicles.keys()):
                    DG.nodes[now_lane]['outnum'].append(outvehicles[now_lane])

            for k, v in invehicles.items():
                invehicles[k]=0
                outvehicles[k]=0


            ########## for FDTG case study
            for now_lane in DG.nodes.keys():
                DG.nodes[now_lane]['in_vehicle_hops'].append([0,0,0,0,0,0,0,0])

            minute_snapshot = vehicles




def calc_ratio(time_interval,pre_vehicles,vehicles,step,calc_outvehicles,DG):


    if (step > 0):
        for vehicle, info in vehicles.items():
            if (vehicle not in pre_vehicles.keys()):
                continue
            if (info['route'] != pre_vehicles[vehicle]['route']):
                tar_lane = int(info['drivable'][0])
                now_lane = int(pre_vehicles[vehicle]['drivable'][0])
                if((now_lane,tar_lane) in calc_outvehicles.keys()):
                    calc_outvehicles[(now_lane, tar_lane)] += 1
        if (step + 1) % time_interval == 0:
            for now_lane in DG.nodes.keys():
                total_num = 0
                for suc in list(DG.successors(now_lane)):
                    total_num += calc_outvehicles[(now_lane, suc)]
                for suc in list(DG.successors(now_lane)):
                    if (total_num != 0):
                        DG.edges[(now_lane, suc)]['ratio'].append(calc_outvehicles[(now_lane, suc)] / total_num)
                    else:
                        DG.edges[(now_lane, suc)]['ratio'].append(calc_outvehicles[(now_lane, suc)])
                    calc_outvehicles[(now_lane, suc)] = 0

        pre_vehicles = vehicles

def calc_greensec(time_interval,phases,step,DG,lane2greesec):
    phase_map = [
        [-1, -1, -1, -1],
        [0, 0, 2, 0],
        [0, 1, 2, 1],
        [1, 0, 3, 0],
        [1, 1, 3, 1],
        [0, 0, 0, 1],
        [1, 0, 1, 1],
        [2, 0, 2, 1],
        [3, 0, 3, 1]
    ]
    for k in DG.nodes.keys():
        if ((step + 1) % time_interval == 0):
            if(DG.nodes[k]['have_signal'] == False):
                DG.nodes[k]['greensec'].append(time_interval)
    for agent, phase in phases.items():
        in_roads = agents[agent][:4]
        avaliable_road = [in_roads[phase_map[phase][0]], in_roads[phase_map[phase][2]]]
        avaliable_lane = [avaliable_road[0]*100 + phase_map[phase][1], avaliable_road[1] * 100 + phase_map[phase][3]]
        if(avaliable_lane[0] != -100):
            lane2greesec[avaliable_lane[0]] +=1
        if(avaliable_lane[1] != -100):
            lane2greesec[avaliable_lane[1]] +=1
    if((step + 1)% time_interval == 0):
        for k in lane2greesec.keys():
            if(DG.nodes[k]['have_signal'] == False):
                continue
            if(k%100 == 2):
                DG.nodes[k]['greensec'].append(time_interval)
            else:
                DG.nodes[k]['greensec'].append(lane2greesec[k])
                lane2greesec[k] = 0
def calc_demand(time_interval,step,vehicles,roads,intersections,cur_demand,DG):
    for k, v in vehicles.items():
        if(len(v['route']) < 2):
            continue
        road = int(v['road'][0])
        speed_limit = roads[road]['speed_limit']
        remain_time = (time_interval - step % time_interval)
        need_time = (roads[road]['length'] - float(v['distance'][0])) / speed_limit
        if(need_time < remain_time):
            if(len(v['route']) == 2):# 下一条路三条lane等概率
                for lane_id in range(3):
                    tar_lane = int(v['route'][1]) * 100 + lane_id
                    cur_demand[tar_lane] += 1/3
            else: # 否则查看是上哪条道
                next_road = int(v['route'][1])
                nnext_road = int(v['route'][2])
                inter = roads[next_road]['end_inter']
                if(intersections[inter]['have_signal'] == False): #没有信号灯
                    for lane_id in range(3):
                        tar_lane = next_road * 100 + lane_id
                        cur_demand[tar_lane] += 1 / 3
                else:
                    tar_lane = -1
                    for lane_id in range(3):
                        next_lane = next_road * 100 + lane_id
                        for suc in DG.successors(next_lane):
                            if(suc//100 == nnext_road):
                                tar_lane = next_lane
                    cur_demand[tar_lane] += 1
    if((step + 1)%time_interval == 0):
        for k,v in DG.nodes.items():
            DG.nodes[k]['demand'].append(cur_demand[k] )
            cur_demand[k] = 0

def check_split_idx(dis):
    if(dis <=30):
        return 0
    elif(dis<=60):
        return 1
    elif(dis<=100):
        return 2
    else:
        return 3
def  calc_target(time_interval,step,vehicles,lane2volume,lane2speed,DG,roads):
    for vehicle,val in vehicles.items():
        cur_lane = int(val['drivable'][0])
        lane2volume[cur_lane] +=1
        lane2speed[cur_lane][0] += float(val['speed'][0])
        lane2speed[cur_lane][1] += 1


    if((step + 1)% time_interval == 0):
        lane2vol = {}
        ## 瞬时volume
        for vehicle, val in vehicles.items():
            cur_lane = int(val['drivable'][0])
            if(cur_lane not in lane2vol.keys()):
                lane2vol[cur_lane] = 0
            lane2vol[cur_lane] +=1

        for k , v in DG.nodes.items():
            DG.nodes[k]['volume_split'].append([0,0,0,0])

        for vehicle, val in vehicles.items():
            cur_lane = int(val['drivable'][0])
            cur_distance = float(val['distance'][0])
            cur_road = int(val['road'][0])
            tot_length = roads[cur_road]['length']

            remain_length = tot_length - cur_distance

            idx = check_split_idx(remain_length)
            # print(cur_lane,idx)
            DG.nodes[cur_lane]['volume_split'][-1][idx] +=1


        for k in DG.nodes.keys():
            if(k in lane2speed.keys() and lane2speed[k][1] == 0):
                avg_speed = -1
            else:
                avg_speed = lane2speed[k][0] / lane2speed[k][1]
            # 瞬时
            if(k in lane2vol.keys()):
                now_vol = lane2vol[k]

            # # 平均
            # if(k in lane2volume.keys()):
            #     now_vol = lane2volume[k] / time_interval

            else:
                now_vol = 0

            DG.nodes[k]['target'].append([now_vol,avg_speed])
            DG.nodes[k]['occupancy'].append(lane2volume[k])
            lane2volume[k] = 0
            lane2speed[k] = [0,0]
            lane2vol[k] = 0


        # for k in lane2volume.keys():
        #     if(lane2speed[k][1] == 0):
        #         avg_speed = -1
        #     else:
        #         avg_speed = lane2speed[k][0] / lane2speed[k][1]
        #     DG.nodes[k]['target'].append([lane2volume[k] / time_interval,avg_speed])
        #     lane2volume[k] = 0
        #     lane2speed[k] = [0,0]

if __name__ == "__main__":
    ######################
    #gym
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
    for k in observations:
        agent_id_list.append(int(k.split('_')[0]))
    agent_id_list = list(set(agent_id_list))
    agent = TestAgent()
    # agent = FixedTimeAgent(fixed_time=60)
    agent.load_agent_list(agent_id_list)
    simulator_configs = read_config(simulator_cfg_file)
    mx_step = simulator_configs['max_time_epoch']
    roadnet_path = Path(simulator_configs['road_file_addr'])
    intersections, roads, agents = process_roadnet(roadnet_path)
    agent.load_roadnet(intersections, roads, agents)
    done = False

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
    setting = "hangzhou_scale_5"
    suf = 'instant_volume'
    while not done:
        actions = {}

        all_info = {
            'observations':observations,
            'info':infos,
            'cur_time': step
        }
        # logger.info("step : {}, avg_tt : {}".format(step, env.eng.get_average_travel_time()))
        actions = agent.act(all_info)
        observations, rewards, dones, infos = env.step(actions)

        # collect data
        cur_vehicle_info = {}
        eng = env.eng
        vehicles = eng.get_vehicles()
        for vehicle in vehicles:
            cur_vehicle_info[vehicle] = eng.get_vehicle_info(vehicle)

        cur_phases = {}
        for agent_id, phase in actions.items():
            cur_phases[int(agent_id)] = int(phase)


        # calc ratio
        calc_ratio(config['time_interval'], pre_vehicles, cur_vehicle_info,step,calc_outvehicles,DG)
        calc_inoutnum(config['time_interval'], pre_vehicles, cur_vehicle_info, step, invehicles,outvehicles,DG,minute_snapshot,invehicles_id)
        pre_vehicles = cur_vehicle_info

        # calc greensec
        calc_greensec(config['time_interval'],cur_phases,step,DG,lane2greesec)

        # calc demand
        calc_demand(config['time_interval'],step,cur_vehicle_info,roads,intersections,cur_demand,DG)

        # calc target
        calc_target(config['time_interval'],step,cur_vehicle_info,lane2volume,lane2speed,DG,roads)

        if((step + 1)% config['time_interval'] == 0):
            minute_snapshot = cur_vehicle_info

        for agent_id in agent_id_list:
            if(dones[agent_id]):
                done = True
        step += 1
        print("{}/{}".format(step,mx_step))
        if((step + 1) % 600 == 0):
            nx.write_gpickle(DG, '{}_graph_processed_straight_withInOut_splitlength_{}.gpickle'.format(setting, suf))

    nx.write_gpickle(DG,'{}_graph_processed_straight_withInOut_splitlength_{}.gpickle'.format(setting,suf))


    print('-----------------------')

