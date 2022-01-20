import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import citypb

import numpy as np
import pickle
import random
import time
import json
import torch
from wrapper import Wrapper

def main():
    roadnet_file = './data/roadnet_hangzhou.txt'
    flow_file = './data/flow_hangzhou_old.txt'

    wrapper = Wrapper(roadnet_file, flow_file, None)
    wrapper.dist()
        
    pass


if __name__ == '__main__':
    main()
