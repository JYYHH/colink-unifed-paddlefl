import os
import json
import sys
import subprocess
import tempfile
from typing import List
from time import sleep
import collections
import time

import colink as CL

from unifed.frameworks.paddlefl.util import store_error, store_return, GetTempFileName, get_local_ip

pop = CL.ProtocolOperator(__name__)
UNIFED_TASK_DIR = "unifed:task"

def load_config_from_param_and_check(param: bytes):
    unifed_config = json.loads(param.decode())
    framework = unifed_config["framework"]
    assert framework == "paddlefl"
    deployment = unifed_config["deployment"]
    if deployment["mode"] != "colink":
        raise ValueError("Deployment mode must be colink")
    return unifed_config

def simulate_workload(role, participant_id, server_ip):
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
                "fl_master.py",
                server_ip
            ]
        )

        subprocess.Popen(
            [
                "python3.8",  
                "-u",
                "fl_scheduler.py",
                server_ip
            ]
        )

        sleep(5)

        subprocess.Popen(
            [
                "python3.8",  
                "-u",
                "fl_server.py",
                f"{participant_id}",
                server_ip
            ]
        )
        sleep(2)

        print("Server OK JOB")



    elif role == 'client':
        
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

        print("Leaf_utils over")

        # os.system("unset http_proxy")
        # os.system("unset https_proxy")
        # os.system("ps -ef | grep -E fl_ | grep -v grep | awk '{print $2}' | xargs kill -9")
        os.system("mkdir logs")

        subprocess.run(
            [
                "python3.8",  
                "fl_master.py",
                server_ip
            ]
        )

        trainer = subprocess.Popen(
            [
                "python3.8",  
                "fl_trainer.py",
                participant_id,
                server_ip
            ],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        stdout, stderr = trainer.communicate()
        print(f"STDOUT = \n    {stdout}")
        print(f"STDERR = \n    {stderr}")
    else:
        raise ValueError(f'Invalid role {role}')
    # or, alternatively
    # with open(log_path, 'w') as f:
    #     f.write(f"Some log for {role} here.")
    print('Simulated workload here end.')

def run_external_process_and_collect_result(cl: CL.CoLink, participant_id,  role: str, server_ip: str):
    # note that here, you don't have to create temp files to receive output and log
    # you can also expect the target process to generate files and then read them

    # start training procedure

    # process = subprocess.Popen(
    #     [
    #         "unifed-paddlefl-workload",  
    #         # takes 4 args: mode(client/server), participant_id, output, and logging destination
    #         role,
    #         str(participant_id),
    #         server_ip
    #     ],
    #     stdout=subprocess.PIPE, 
    #     stderr=subprocess.PIPE
    # )

    simulate_workload(role, str(participant_id), server_ip)

    # print(f"Writing to {temp_output_filename} and {temp_log_filename}...")
    # with open(temp_output_filename, 'w') as f:
    #     f.write(f"Some output for {role} here.")
    # if role == "server":
    #     with open(temp_log_filename, 'w') as f:
    #         with open("./log/0.log", 'r') as f2:
    #             f.write(f2.read())

    # gather result
    # with open(temp_output_filename, "rb") as f:
    #     output = f.read()
    # cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", output)
    # with open(temp_log_filename, "rb") as f:
    #     log = f.read()
    # cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", log)

    return json.dumps({
        "server_ip": server_ip,
        "stdout": "None",
        "stderr": "None",
        "returncode": 0,
    })


@pop.handle("unifed.paddlefl:server")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_server(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    with open("config.json", "w") as write_file:
        json.dump(unifed_config, write_file)
    # for certain frameworks, clients need to learn the ip of the server
    # in that case, we get the ip of the current machine and send it to the clients
    server_ip = get_local_ip()
    # server_ip = "127.0.0.1"
    cl.send_variable("server_ip", server_ip, [p for p in participants if p.role == "client"])
    client_list = [p for p in participants if p.role == "client"]
    for idx, client in enumerate(client_list):
        cl.send_variable("client_true_id", str(idx + 1), [client])
    # run external program
    # participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    participant_id = 0
    ret = run_external_process_and_collect_result(cl, participant_id, "server", server_ip)
    cl.send_variable("server_status", "OK", [p for p in participants if p.role == "client"])
    cl.recv_variable("train_over", client_list[0]).decode()
    print("SERVER confirms Train Over")
    
    with open("/test/log/0.log", 'r') as f2:
        print(
            "SERVER is Writing the FINAL LOGGING file"
        )
        # f.write(f2.read())
        # print(f2.read())
        cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", f2.read())
        return json.dumps({
            "server_ip": server_ip,
            "stdout": "None",
            "stderr": "None",
            "returncode": 0,
        })

    return ret


@pop.handle("unifed.paddlefl:client")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_client(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    print("INTO CLIENT")
    unifed_config = load_config_from_param_and_check(param)
    with open("config.json", "w") as write_file:
        json.dump(unifed_config, write_file)
    # get the ip of the server
    server_in_list = [p for p in participants if p.role == "server"]
    assert len(server_in_list) == 1
    p_server = server_in_list[0]

    server_ip = cl.recv_variable("server_ip", p_server).decode()
    participant_id = int(cl.recv_variable("client_true_id", p_server).decode())
    server_setup = cl.recv_variable("server_status", p_server).decode()
    print(f"CLIENT {participant_id} CAN RUN")
    # run external program
    # participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    ret = run_external_process_and_collect_result(cl, participant_id, "client", server_ip)
    
    if participant_id == 1:
        cl.send_variable("train_over", "true", [p_server])
    # send training and testing over
