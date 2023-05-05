import paddle_fl.paddle_fl as fl
import paddle.fluid as fluid
from paddle_fl.paddle_fl.core.server.fl_server import FLServer
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
import sys
ip_addr = sys.argv[2]

server = FLServer()
server_id = int(sys.argv[1])
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_server_job(job_path, server_id)
job._scheduler_ep = ip_addr + ":9091"  # IP address for scheduler
server.set_server_job(job)
server._current_ep = ip_addr + ":8981"  # IP address for server
server.start()
