import os
import json
import sys
import subprocess
import tempfile
from typing import List

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

def run_external_process_and_collect_result(cl: CL.CoLink, participant_id,  role: str, server_ip: str):
    with GetTempFileName() as temp_log_filename, \
        GetTempFileName() as temp_output_filename:
        # note that here, you don't have to create temp files to receive output and log
        # you can also expect the target process to generate files and then read them

        # start training procedure
        process = subprocess.Popen(
            [
                "unifed-paddlefl-workload",  
                # takes 4 args: mode(client/server), participant_id, output, and logging destination
                role,
                str(participant_id),
                server_ip
            ],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )

        # print(f"Writing to {temp_output_filename} and {temp_log_filename}...")
        # with open(temp_output_filename, 'w') as f:
        #     f.write(f"Some output for {role} here.")
        # with open(temp_log_filename, 'w') as f:
        #     with open(f"./log/{participant_id}.log", 'r') as f2:
        #         f.write(f2.read())

        # gather result
        stdout, stderr = process.communicate()
        returncode = process.returncode
        # with open(temp_output_filename, "rb") as f:
        #     output = f.read()
        # cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", output)
        # with open(temp_log_filename, "rb") as f:
        #     log = f.read()
        # cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", log)

        if role == "server":
            cl.send_variable("server_status", "OK")

        return json.dumps({
            "server_ip": server_ip,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": returncode,
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
    cl.send_variable("server_ip", server_ip, [p for p in participants if p.role == "client"])
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "server", server_ip)


@pop.handle("unifed.paddlefl:client")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_client(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    with open("config.json", "w") as write_file:
        json.dump(unifed_config, write_file)
    # get the ip of the server
    server_in_list = [p for p in participants if p.role == "server"]
    assert len(server_in_list) == 1
    p_server = server_in_list[0]
    server_ip = cl.recv_variable("server_ip", p_server).decode()

    server_setup = cl.recv_variable("server_status", p_server).decode()
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "client", server_ip)

    # send training and testing over
