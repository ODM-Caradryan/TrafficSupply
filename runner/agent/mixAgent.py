from pathlib import Path
import pickle
import gym
import numpy as np

class MixAgent():
    def __init__(self, agent_pool, agent_type, turn_cycle=900):
        self.agent_pool = agent_pool
        self.turn_cycle = turn_cycle
        self.agent_type = [x if x < len(agent_pool) and x >= 0 else 0 for x in agent_type]
        
        # # print("MAX PRESSURE")
    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self, agent_list):
        for agent in self.agent_pool:
            agent.load_agent_list(agent_list)

    def load_roadnet(self,intersections,roads,agents):
        for agent in self.agent_pool:
            agent.load_roadnet(intersections,roads,agents)


    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        cur_time = obs['cur_time']
        agent_type = self.agent_type[int(cur_time / self.turn_cycle) % len(self.agent_type)]
        # here obs contains all of the observations and infos
        
        actions = {}
        for agent_idx, agent in enumerate(self.agent_pool):
            spare_actions = agent.act(obs)
            actions = spare_actions if agent_idx == int(agent_type) else actions
            
        return actions