# inner_epoch / loss_func

import sys
from time import sleep
import subprocess
import json
import collections
import time 
import os 


def simulate_workload():
    argv = sys.argv

    if len(argv) != 3:
        raise ValueError(f'Invalid arguments. Got {argv}')
    role, participant_id = argv[1:3]
    
    print('Simulated workload here begin.')

    if role == 'server':
        subprocess.run(
            [
                "python3.8",  
                "download.py"
            ]
        )

        subprocess.run(
            [
                "python3.8",  
                "leaf_utils.py"
            ]
        )

        os.system("unset http_proxy")
        os.system("unset https_proxy")
        os.system("ps -ef | grep -E fl_ | grep -v grep | awk '{print $2}' | xargs kill -9")
        os.system("mkdir logs")

        subprocess.run(
            [
                "python3.8",  
                "fl_master.py"
            ]
        )

        subprocess.Popen(
            [
                "python3.8",  
                "-u",
                "fl_scheduler.py",
            ]
        )

        sleep(5)

        subprocess.Popen(
            [
                "python3.8",  
                "-u",
                "fl_server.py",
                f"{participant_id}",
            ]
        )
        sleep(2)



    elif role == 'client':
        
        # subprocess.run(
        #     [
        #         "python3.8",  
        #         "download.py"
        #     ]
        # )

        # subprocess.run(
        #     [
        #         "python3.8",  
        #         "leaf_utils.py"
        #     ]
        # )

        # os.system("unset http_proxy")
        # os.system("unset https_proxy")
        # os.system("ps -ef | grep -E fl_ | grep -v grep | awk '{print $2}' | xargs kill -9")
        # os.system("mkdir logs")

        # # subprocess.run(
        # #     [
        # #         "python3.8",  
        # #         "fl_master.py"
        # #     ]
        # # )

        subprocess.Popen(
            [
                "python3.8",  
                "-u",
                "fl_trainer.py",
                participant_id
            ]
        )

    else:
        raise ValueError(f'Invalid role {role}')

    simulate_logging(participant_id, role)
    # or, alternatively
    # with open(log_path, 'w') as f:
    #     f.write(f"Some log for {role} here.")
    print('Simulated workload here end.')


def simulate_logging(participant_id, role):
    # source: https://github.com/AI-secure/FLBenchmark-toolkit/blob/166a7a42a6906af1190a15c2f9122ddaf808f39a/tutorials/logging/run_fl.py
    # if role == 'server':
    #     pass
    # elif role == 'client':
    #     pass
    # else:
    #     raise ValueError(f'Invalid role {role}')
    pass
