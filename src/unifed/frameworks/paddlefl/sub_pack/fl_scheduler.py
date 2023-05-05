from paddle_fl.paddle_fl.core.scheduler.agent_master import FLScheduler
import json
config = json.load(open('config.json', 'r'))
import pickle
import sys
ip_addr = sys.argv[1]

sample_num = config["training"]["client_per_round"]
server_num = 1
scheduler = FLScheduler(sample_num, server_num, port=9091)
scheduler.set_sample_worker_num(sample_num)
scheduler.init_env()
print("init env done.")
model_size = pickle.load(open("model_size.pkl", "rb"))
scheduler.start_fl_training(model_size, sample_num)
