import sys

from unifed.frameworks.paddlefl import protocol
from unifed.frameworks.paddlefl.workload_sim import *

def run_protocol():
    print('Running protocol...')
    protocol.pop.run()  # FIXME: require extra testing here

