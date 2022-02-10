import os, sys
# import citypb

import numpy as np
import pickle
import random
import time
import json
# import torch
from wrapper import Wrapper

def main():
    roadnet_file = './data/roadnet_round2.txt'
    flow_file = './data/flow_round2.txt'
    target_file = './data/flow_round2_test.txt'

    wrapper = Wrapper(roadnet_file, flow_file, None)
    # wrapper.dist()
    wrapper.generate_flow(target_file=target_file)
        
    pass


if __name__ == '__main__':
    main()
