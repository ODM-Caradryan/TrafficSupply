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
        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        self.phase_lane_map_out = [[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
                                   [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24],
                                   [16, 17, 18, 19, 20, 21], [19, 20, 21, 22, 23, 24],
                                   [13, 14, 15, 22, 23, 24], [13, 14, 15, 16, 17, 18]]
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
        unavailable_phases = self.get_unavailable_phases(lane_vehicle_num)
        # if len(unavailable_phases) > 0:
        #     # print("unavailable_phases: ", unavailable_phases)
        available_phases = list(set(list(range(1, 9))) - set(unavailable_phases))
        action_id = available_phases[(cur_time / fixed_time) % len(available_phases)]
        # # print(max_pressure_id)
        return action_id



    def get_unavailable_phases(self, lane_vehicle_num):
        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        unavailable_phases = []
        not_exist_lanes = []
        for i in range(1, 25):
            if lane_vehicle_num[i] < 0:
                not_exist_lanes.append(i)
        for lane in not_exist_lanes:
            for phase_id, in_lanes in enumerate(self.phase_lane_map_in):
                phase_id += 1
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
        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        self.phase_lane_map_out = [[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
                                   [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24],
                                   [16, 17, 18, 19, 20, 21], [19, 20, 21, 22, 23, 24],
                                   [13, 14, 15, 22, 23, 24], [13, 14, 15, 16, 17, 18]]
        # # print("MAX PRESSURE")
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
        for i in range(8):
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
        pressures = self.get_phase_pressures(lane_vehicle_num)
        unavailable_phases = self.get_unavailable_phases(lane_vehicle_num)
        # if len(unavailable_phases) > 0:
        #     # print("unavailable_phases: ", unavailable_phases)

        max_pressure_id = np.argmax(pressures) + 1
        while (max_pressure_id in unavailable_phases):
            pressures[max_pressure_id - 1] -= 999999
            max_pressure_id = np.argmax(pressures) + 1
        # # print(max_pressure_id)
        return max_pressure_id



    def get_unavailable_phases(self, lane_vehicle_num):
        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        unavailable_phases = []
        not_exist_lanes = []
        for i in range(1, 25):
            if lane_vehicle_num[i] < 0:
                not_exist_lanes.append(i)
        for lane in not_exist_lanes:
            for phase_id, in_lanes in enumerate(self.phase_lane_map_in):
                phase_id += 1
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
        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        self.phase_lane_map_out = [[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
                                   [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24],
                                   [16, 17, 18, 19, 20, 21], [19, 20, 21, 22, 23, 24],
                                   [13, 14, 15, 22, 23, 24], [13, 14, 15, 16, 17, 18]]
        
        self.volumes = {}
        self.split_end_time = 0
        
        # # print("MAX PRESSURE")
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


    def get_phase_volumes(self, lane_vehicle_num):
        volumes = []
        for i in range(4, 8):
            in_lanes = self.phase_lane_map_in[i]
            volume = 0
            for in_lane in in_lanes:
                volume = lane_vehicle_num[in_lane] if volume < lane_vehicle_num[in_lane] else volume
            volumes.append(volume)
        # # print("pressures: ", pressures)
        return volumes
    
    def get_cycle_length(self, volumes, cur_time):
        if cur_time == 0:
            return 200 # initialize the cycle as 200s
        
        # determine the cycle
        total_vol =  np.sum(volumes)
        h = 2.45
        tL = 7
        PHF = 1
        vc = 1
        N = 8

        max_allowed_vol = 3600 / h * PHF * vc
        if total_vol/max_allowed_vol > 0.95:
            cycle_length = N * tL / (1 - 0.95)
        else:
            cycle_length = N * tL / (1 - total_vol / max_allowed_vol)
        
        return cycle_length
    
    def round_up(self, x, min_phase_time=10, b=5):
        round_x = (b * np.ceil(x.astype(float) / b)).astype(int)
        round_x[np.where(round_x < min_phase_time and round_x >= 0)] = min_phase_time
        return round_x

    def get_action(self, lane_vehicle_num, cur_time, agent_id):
        if agent_id not in self.volumes.keys():
            self.volumes[agent_id] = np.zeros((4,))
        
        volumes = self.get_phase_volumes(lane_vehicle_num)
        self.volumes[agent_id] = self.volumes[agent_id] + np.array(volumes)
        
        if cur_time < self.split_end_time:
            for tmp_id, phase_time in enumerate(self.phase_split):
                # an unavailable phase would be like [50, 150, 150, 200], ignored
                if (cur_time + self.cycle_length - self.split_end_time) % self.cycle_length < phase_time:
                    # phase = phase_id + 1
                    phase = tmp_id + 5 # use 5-8
                    return phase
        else:            
            # determine raw cycle length
            self.cycle_length = self.get_cycle_length(self.volumes[agent_id], cur_time)
            
            if np.sum(self.volumes[agent_id]) != 0:
                self.phase_split = np.copy(self.volumes[agent_id])\
                          / np.sum(self.volumes[agent_id]) \
                          * self.cycle_length
            else:
                self.phase_split = np.full(shape=(np.shape(self.volumes[agent_id])[0],),\
                                    fill_value=1/len(self.volumes)) \
                                    * self.cycle_length
            self.phase_split = self.round_up(self.phase_split)
                        
            # convert the split into pdf
            for phase_id in range(1, 4):
                self.phase_split[phase_id] = self.phase_split[phase_id] + self.phase_split[phase_id-1]
            
            self.cycle_length = self.phase_split[-1] # update cycle time
            self.volumes[agent_id] = np.zeros((4,)) # clear the volumes record
            self.split_end_time = cur_time + self.cycle_length # update end time
            
            # begin new cycle
            for tmp_id, phase_time in enumerate(self.phase_split):
                # an unavailable phase would be like [50, 150, 150, 200], ignored
                if (cur_time + self.cycle_length - self.split_end_time) % self.cycle_length < phase_time:
                    # phase = phase_id + 1
                    phase = tmp_id + 5 # use 5-8
                    return phase
        


    def get_unavailable_phases(self, lane_vehicle_num):
        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        unavailable_phases = []
        not_exist_lanes = []
        for i in range(1, 25):
            if lane_vehicle_num[i] < 0:
                not_exist_lanes.append(i)
        for lane in not_exist_lanes:
            for phase_id, in_lanes in enumerate(self.phase_lane_map_in):
                phase_id += 1
                if lane in in_lanes and phase_id not in unavailable_phases:
                    unavailable_phases.append(phase_id)

        return unavailable_phases
        # return [5, 6, 7, 8]

    def update_volumes(self, env):
        pass

    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        # here obs contains all of the observations and infos
        observations = obs['observations']
        info = obs['info']
        cur_time = obs['cur_time']
        env = obs['env']
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
        self.update_volumes(env)

        for agent in self.agent_list:
            # select the now_step
            for k,v in observations_for_agent[agent].items():
                now_step = v[0]
                break
            lane_vehicle_num = observations_for_agent[agent]["lane_vehicle_num"]
            # print("agent id: ", agent)
            # print("lane vehicle: ", lane_vehicle_num)
            action = self.get_action(lane_vehicle_num, cur_time, agent)
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