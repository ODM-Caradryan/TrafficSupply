from pathlib import Path
import pickle
import gym
import numpy as np

class FixedTimeAgent():
    def __init__(self, fixed_time):
        self.fixed_time = fixed_time
        
        self.now_phase = {}
        self.green_sec = 20
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}
        # self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        self.phase_lane_map_in = [[2, 1], [5, 4], [8, 7], [11, 10]]
        #self.phase_lane_map_out = [[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
        #                           [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24],
        #                           [16, 17, 18, 19, 20, 21], [19, 20, 21, 22, 23, 24],
        #                           [13, 14, 15, 22, 23, 24], [13, 14, 15, 16, 17, 18]]
        # # print("MAX PRESSURE")
    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self, agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)

    def load_roadnet(self,intersections,roads,agents):
        # in_roads = []
        # for agent, agent_roads in agents:
        #     in_roads = agent_roads[:4]
        #     now_phase = dict.fromkeys(range(9))
        #     now_phase[0] = []
        #     in_roads[0]
        pass


    def get_action(self, lane_vehicle_num, cur_time, fixed_time):
        # unavailable_phases = self.get_unavailable_phases(lane_vehicle_num)
        # if len(unavailable_phases) > 0:
        #     # print("unavailable_phases: ", unavailable_phases)
        
        # available_phases = list(set(list(range(1, 9))) - set(unavailable_phases))
        available_phases = [5, 6, 7, 8]
        action_id = available_phases[int(cur_time / fixed_time) % len(available_phases)]
        # # print(max_pressure_id)
        # print(action_id)
        return action_id


    '''
    def get_unavailable_phases(self, lane_vehicle_num):
        # self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        self.phase_lane_map_in = [[2, 1], [5, 4], [8, 7], [11, 10]]
        unavailable_phases = []
        not_exist_lanes = []
        for i in range(1, 25):
            if lane_vehicle_num[i] < 0:
                not_exist_lanes.append(i)
        for lane in not_exist_lanes:
            for phase_id, in_lanes in enumerate(self.phase_lane_map_in):
                # phase_id += 1
                phase_id += 5
                if lane in in_lanes and phase_id not in unavailable_phases:
                    unavailable_phases.append(phase_id)

        return unavailable_phases
        # return [5, 6, 7, 8]
    '''

    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        # here obs contains all of the observations and infos
        observations = obs['observations']
        info = obs['info']
        cur_time = obs['cur_time']
        actions = {}


        # preprocess observations
        # a simple fixtime agent
        observations_for_agent = {}
        for key,val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_')+1:]
            if(observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val

        for agent in self.agent_list:
            # select the now_step
            for k,v in observations_for_agent[agent].items():
                now_step = v[0]
                break
            lane_vehicle_num = observations_for_agent[agent]["lane_vehicle_num"]
            # print("agent id: ", agent)
            # print("lane vehicle: ", lane_vehicle_num)
            action = self.get_action(lane_vehicle_num, cur_time, self.fixed_time)
            # print("action: ", action)

            step_diff = now_step - self.last_change_step[agent]
            if (step_diff >= self.green_sec):
                self.now_phase[agent] = action
                self.last_change_step[agent] = now_step

            actions[agent] = self.now_phase[agent]
            # print("phase: ", actions[agent])
            # print("phase available lane: ", self.phase_lane_map_in[actions[agent]-1])
            # print("________")

        return actions
    
class MPAgent():
    def __init__(self):
        self.now_phase = {}
        self.green_sec = 20
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}
        #self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        #self.phase_lane_map_out = [[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
        #                           [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24],
        #                           [16, 17, 18, 19, 20, 21], [19, 20, 21, 22, 23, 24],
        #                           [13, 14, 15, 22, 23, 24], [13, 14, 15, 16, 17, 18]]
        # # print("MAX PRESSURE")
        self.phase_lane_map_in = [[2, 1], [5, 4], [8, 7], [11, 10]]
        self.phase_lane_map_out = [[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
                                   [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24]]
    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)

    def load_roadnet(self,intersections,roads,agents):
        # in_roads = []
        # for agent, agent_roads in agents:
        #     in_roads = agent_roads[:4]
        #     now_phase = dict.fromkeys(range(9))
        #     now_phase[0] = []
        #     in_roads[0]
        pass

    ################################


    def get_phase_pressures(self, lane_vehicle_num):
        pressures = []
        for i in range(4):   # use 4-7(5-8)
            in_lanes = self.phase_lane_map_in[i]
            out_lanes = self.phase_lane_map_out[i]
            pressure = 0
            for in_lane in in_lanes:
                pressure += lane_vehicle_num[in_lane] * 3
            for out_lane in out_lanes:
                pressure -= lane_vehicle_num[out_lane]
            pressures.append(pressure)
        # # print("pressures: ", pressures)
        return pressures

    def get_action(self, lane_vehicle_num):
        # zero all non-existing phases
        lane_vehicle_num = [x if x >= 0 else 0 for x in lane_vehicle_num]
            
        pressures = self.get_phase_pressures(lane_vehicle_num)
        # unavailable_phases = self.get_unavailable_phases(lane_vehicle_num)
        # if len(unavailable_phases) > 0:
        #     # print("unavailable_phases: ", unavailable_phases)

        max_pressure_id = np.argmax(pressures) + 5  # turn the idx to phase id (use 5-8)
        '''
        while (max_pressure_id in unavailable_phases):
            pressures[max_pressure_id - 1] -= 999999
            max_pressure_id = np.argmax(pressures) + 1
        '''
        # # print(max_pressure_id)
        return max_pressure_id



    def get_unavailable_phases(self, lane_vehicle_num):
        self.phase_lane_map_in = [[2, 1], [5, 4], [8, 7], [11, 10]]
        unavailable_phases = []
        not_exist_lanes = []
        for i in range(1, 25):
            if lane_vehicle_num[i] < 0:
                not_exist_lanes.append(i)
        for lane in not_exist_lanes:
            for phase_id, in_lanes in enumerate(self.phase_lane_map_in):
                # phase_id += 1
                phase_id += 5
                if lane in in_lanes and phase_id not in unavailable_phases:
                    unavailable_phases.append(phase_id)

        return unavailable_phases
        # return [5, 6, 7, 8]


    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        # here obs contains all of the observations and infos
        observations = obs['observations']
        info = obs['info']
        actions = {}


        # preprocess observations
        # a simple fixtime agent
        observations_for_agent = {}
        for key,val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_')+1:]
            if(observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val

        for agent in self.agent_list:
            # select the now_step
            for k,v in observations_for_agent[agent].items():
                now_step = v[0]
                break
            lane_vehicle_num = observations_for_agent[agent]["lane_vehicle_num"]
            # print("agent id: ", agent)
            # print("lane vehicle: ", lane_vehicle_num)
            action = self.get_action(lane_vehicle_num)
            # print("action: ", action)

            step_diff = now_step - self.last_change_step[agent]
            if (step_diff >= self.green_sec):
                self.now_phase[agent] = action
                self.last_change_step[agent] = now_step

            actions[agent] = self.now_phase[agent]
            # print("phase: ", actions[agent])
            # print("phase available lane: ", self.phase_lane_map_in[actions[agent]-1])
            # print("________")

        return actions
    

class FormulaAgent():
    def __init__(self):
        self.now_phase = {}
        self.green_sec = 20
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}
        #self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        #self.phase_lane_map_out = [[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
        #                           [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24],
        #                           [16, 17, 18, 19, 20, 21], [19, 20, 21, 22, 23, 24],
        #                           [13, 14, 15, 22, 23, 24], [13, 14, 15, 16, 17, 18]]
        # # print("MAX PRESSURE")
        self.phase_lane_map_in = [[2, 1], [5, 4], [8, 7], [11, 10]]
        self.phase_lane_map_out = [[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
                                   [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24]]
        
        self.lane_vehicles = {}
        self.lane_volumes = {}
        self.init_flags = {}
        
        self.clock_time = 0
        self.end_time = 0
        
        
        # # print("MAX PRESSURE")
    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self, agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)

    def load_roadnet(self,intersections,roads,agents):
        
        self.intersections = intersections
        self.roads = roads
        self.agents = agents
        
        # in_roads = []
        # for agent, agent_roads in agents:
        #     in_roads = agent_roads[:4]
        #     now_phase = dict.fromkeys(range(9))
        #     now_phase[0] = []
        #     in_roads[0]
        pass

    ################################
    
    def get_cycle_length(self, agent_id, phase_volume):
        if agent_id not in self.init_flags.keys():
            self.init_flags[agent_id] = True
            return 300 # initialize the cycle as 300s
        
        # determine the cycle
        '''
        total_vol = np.sum(phase_volume)
        h = 2.45
        tL = 7
        PHF = 1
        vc = 1
        N = 8

        max_allowed_vol = self.end_time / h * PHF * vc
        if total_vol/max_allowed_vol > 0.95:
            cycle_length = N * tL / (1 - 0.95)
        else:
            cycle_length = N * tL / (1 - total_vol / max_allowed_vol)
        '''
        cycle_length = np.random.randint(180, 300)
        
        return cycle_length if cycle_length > 60 else 60
    

    def get_action(self, agent_id):
        if self.clock_time < self.end_time and agent_id in self.init_flags.keys():
            for tmp_id, phase_time in enumerate(self.phase_time):
                # an unavailable phase would be like [50, 150, 150, 200], ignored
                if self.clock_time < phase_time:
                    # phase = phase_id + 1
                    phase = tmp_id + 5 # use 5-8
                    self.clock_time += 1 # step

                    return phase
        else:            
            # determine new cycle length
            phase_volume = self.get_phase_volume(agent_id)
            self.end_time = self.get_cycle_length(agent_id, phase_volume)
            # print(phase_volume, self.end_time)
            if np.sum(phase_volume) != 0:
                self.phase_time = np.copy(phase_volume)\
                          / np.sum(phase_volume) \
                          * self.end_time
            else:
                self.phase_time = np.full(shape=(np.shape(phase_volume)[0],),\
                                    fill_value=1/len(phase_volume)) \
                                    * self.end_time
                                    
                        
            # convert the split into pdf
            for phase_id in range(1, 4):
                self.phase_time[phase_id] = self.phase_time[phase_id] + self.phase_time[phase_id-1]
            
            self.lane_vehicles = {}
            self.lane_volumes = {} # clear the volumes record
            self.clock_time = 0
                        
            # begin new cycle
            for tmp_id, phase_time in enumerate(self.phase_time):
                # an unavailable phase would be like [50, 150, 150, 200], ignored
                if self.clock_time < phase_time:
                    # phase = phase_id + 1
                    phase = tmp_id + 5 # use 5-8
                    self.clock_time += 1
                    return phase
                

    def update_volumes(self, eng):
        lane_vehicles = eng.get_lane_vehicles()

        for lane, vehicle_list in lane_vehicles.items():
            if lane not in self.lane_vehicles.keys():
                self.lane_vehicles[lane] = []
            if lane not in self.lane_volumes.keys():
                self.lane_volumes[lane] = 0
            for vehicle in vehicle_list:
                if vehicle not in self.lane_vehicles[lane]:
                    self.lane_volumes[lane] += 1
        self.lane_vehicles = lane_vehicles
        
    def get_phase_volume(self, agent):
        agent_volume = []   # 0-23, tot 24
        for road in self.agents[agent]:
            if road >= 0:
                for lane_idx in range(3):
                    lane_id = road * 100 + lane_idx
                    if lane_id in self.lane_volumes.keys():
                        agent_volume.append(self.lane_volumes[lane_id])
                    else:
                        agent_volume.append(0)
            else:
                for _ in range(3):
                    agent_volume.append(0)
        
        phase_volume = []
        for phase in self.phase_lane_map_in:
            critical_volume = 0
            for lane in phase:  # the range of lane is from 1-24
                critical_volume = max(critical_volume, agent_volume[lane-1])
            phase_volume.append(critical_volume)
                    
        return phase_volume
                
        
    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        # here obs contains all of the observations and infos
        observations = obs['observations']
        info = obs['info']
        eng = obs['eng']
        actions = {}
        
        # preprocess observations
        observations_for_agent = {}
        for key,val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_')+1:]
            if(observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val

        # update volumes
        self.update_volumes(eng)

        for agent in self.agent_list:
            # select the now_step
            for k,v in observations_for_agent[agent].items():
                now_step = v[0]
                break
            action = self.get_action(agent)

            step_diff = now_step - self.last_change_step[agent]
            if (step_diff >= self.green_sec):
                self.now_phase[agent] = action
                self.last_change_step[agent] = now_step

            actions[agent] = self.now_phase[agent]
            # print("phase: ", actions[agent])
            # print("phase available lane: ", self.phase_lane_map_in[actions[agent]-1])
            # print("________")

        return actions