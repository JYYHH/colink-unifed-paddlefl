import paddle
import paddle.fluid as fluid
import paddle_fl.paddle_fl as fl
from paddle_fl.paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory
import my_model
import json 
import pickle
import os

config = json.load(open('config.json', 'r'))

dataset_name = config["dataset"]

try:
    vocab_size = pickle.load(open("vocab.pkl", "rb"))
except:
    vocab_size = 0

regression_dataset = {'student_horizontal'}

in_dim = {'breast_horizontal' : 30,
          'default_credit_horizontal' : 23,
          'give_credit_horizontal' : 10,
          'student_horizontal' : 13,
          'vehicle_scale_horizontal' : 18,
          'femnist' : 28 * 28,
          'reddit' : 10}

out_dim = {'breast_horizontal' : 2,
          'default_credit_horizontal' : 2,
          'give_credit_horizontal' : 2,
          'student_horizontal' : 1,
          'vehicle_scale_horizontal' : 4,
          'femnist' : 62,
          'reddit' : vocab_size}


def get_hidden():
    if config["model"][:3] != "mlp":
        return None
    return [int(x) for x in list(config["model"].split('_'))[1:]]
if config["dataset"] != "reddit":
    model = my_model.get_model(config["model"], in_dim[dataset_name], out_dim[dataset_name], get_hidden())
else:
    model = my_model.lstm(out_dim[dataset_name], 128, 128)

model_byte = model.my_network()
pickle.dump(model_byte, open("model_size.pkl", "wb"))
open('tot_num', 'w').write('tot_num={}'.format(config["training_param"]["client_per_round"]))
open('if_end', 'w').write('if_end=false')

job_generator = JobGenerator()

if config["training_param"]["optimizer"] == 'adam' or config["training_param"]["optimizer"] == 'Adam':
    optimizer = fluid.optimizer.Adam(learning_rate=config["training_param"]["learning_rate"])
elif config["training_param"]["optimizer"] == 'sgd':
    if config["training_param"]["optimizer_param"]["weight_decay"] != 0:
        optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=config["training_param"]["learning_rate"], momentum=config["training_param"]["optimizer_param"]["momentum"], use_nesterov=config["training_param"]["optimizer_param"]["nesterov"], regularization = cn_api_fluid_regularizer_L2Decay)
    else :
        optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=config["training_param"]["learning_rate"], momentum=config["training_param"]["optimizer_param"]["momentum"], use_nesterov=config["training_param"]["optimizer_param"]["nesterov"])#, regularization = cn_api_fluid_regularizer_L2Decay)

job_generator.set_optimizer(optimizer)
job_generator.set_losses([model.loss])
job_generator.set_startup_program(model.startup_program)


job_generator.set_infer_feed_and_target_names(
    [model.inputs.name, model.label.name],
    [model.loss.name, model.predict.name, model.label.name])

build_strategy = FLStrategyFactory()
build_strategy.fed_avg = True
build_strategy.inner_step = config["training_param"]["inner_step"]
strategy = build_strategy.create_fl_strategy()

endpoints = ["127.0.0.1:8981"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=config["training_param"]["client_per_round"], output=output)
