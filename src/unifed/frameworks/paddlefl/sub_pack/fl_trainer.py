from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
import paddle_fl.paddle_fl.dataset.femnist as femnist
import numpy as np
import sys
import paddle
import paddle.fluid as fluid
import logging
import math
import time
from utils import *
from leaf_utils import *
import json 
import flbenchmark.logging
import pickle
import os
from sklearn.metrics import mean_squared_error, roc_auc_score
ip_addr = sys.argv[2]

trainer_id = int(sys.argv[1]) - 1 # trainer id for each guest
    

config = json.load(open('config.json', 'r'))

regression_dataset = {'student_horizontal'}
AUC = ['breast_horizontal', 'default_credit_horizontal', 'give_credit_horizontal',
       'breast_vertical', 'default_credit_vertical', 'give_credit_vertical', ]

LEAF = {'femnist', 'reddit'}

try:
    vocab_size = pickle.load(open("vocab.pkl", "rb"))
except:
    vocab_size = 0

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

is_regression = config["dataset"] in regression_dataset

model_size = pickle.load(open("model_size.pkl", "rb"))

logging.basicConfig(
    filename="test.log",
    filemode="w",
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.DEBUG)


logger = flbenchmark.logging.Logger(id=trainer_id + 1, agent_type='client')

job_path = "fl_job_config"
job = FLRunTimeJob()
print(f"Trainer-ID now is {trainer_id} --------------")
job.load_trainer_job(job_path, trainer_id)
job._scheduler_ep = ip_addr + ":9091"
print(job._target_names)
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(8000 + trainer_id)

# try:
#     place = fluid.CUDAPlace(trainer_id % 4)
# except:
place = fluid.CPUPlace()

trainer.start(place)

print(trainer._step)
test_program = trainer._main_program.clone(for_test=True)

if config["dataset"] != "femnist" or config["model"] != 'lenet':
    if config["dataset"] != "reddit":
        vec = fluid.layers.data(name='vector', shape=[in_dim[config["dataset"]]], dtype='float32')
    else:
        vec = fluid.layers.data(name='vector', shape=[10, 1], dtype='int64')
else:
    vec = fluid.layers.data(name='vector', shape=[-1, 1, 32, 32], dtype='float32')

if is_regression:
    label = fluid.layers.data(name='label', shape=[1], dtype='float32')
elif config["dataset"] != "reddit":
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
else:
    label = fluid.layers.data(name='label', shape=[10, 1], dtype='int64')

feeder = fluid.DataFeeder(feed_list=[vec, label], place=place)


def train_test(train_test_program, train_test_feed, train_test_reader):
    if out_dim[config['dataset']] <= 2:
        predict, target = np.array([]).reshape(-1, out_dim[config['dataset']]), np.array([]).reshape(-1, 1)
        for test_data in train_test_reader():
            acc_np = trainer.exe.run(program=train_test_program,
                                    feed=train_test_feed.feed(test_data),
                                    fetch_list=[job._target_names[1], job._target_names[2]])

            predict = np.vstack((predict, acc_np[0].reshape(-1, out_dim[config['dataset']])))
            target = np.vstack((target, acc_np[1].reshape(-1, 1)))
        
        print('finished!')
        target = target.reshape(-1)

        if config["dataset"] in regression_dataset:
            return mean_squared_error(target, predict.reshape(-1))
        elif config["dataset"] in AUC:
            return roc_auc_score(target, predict[:,1])

    else:
        tot_num, cor_num = 0, 0
        for test_data in train_test_reader():
            acc_np = trainer.exe.run(program=train_test_program,
                                    feed=train_test_feed.feed(test_data),
                                    fetch_list=[job._target_names[1], job._target_names[2]])
            
            predict = acc_np[0].reshape(-1, out_dim[config['dataset']])
            labels = acc_np[1].reshape(-1, 1)
            non_zero_pos = (labels.reshape(-1) != int(-(config['dataset'] != 'reddit')))
            predict = predict[non_zero_pos]
            labels = labels[non_zero_pos]

            tot_num += labels.shape[0]
            cor_num += np.sum(np.argmax(predict, axis = 1).reshape(-1, 1) == labels)

        
        return cor_num / tot_num


epoch_id = 0
step = 0
epoch = config["training"]["global_epochs"]
count_by_step = False
if count_by_step:
    output_folder = "model_node%d" % trainer_id
else:
    output_folder = "model_node%d_epoch" % trainer_id

if config["dataset"] not in LEAF:
    with logger.preprocess_data():
        train_reader = Batch(Train(trainer_id), batch_size = config["training"]["batch_size"])
        test_reader = Batch(Test(), batch_size = config["training"]["batch_size"])
else:
    with logger.preprocess_data():
        test_reader = Batch(LEAFTest(), batch_size = 256)

metric_data = []
test_time = []

with logger.training():
    while not trainer.stop(epoch_id - epoch):
        count = 0
        epoch_id += 1

        if config["dataset"] in LEAF:
            train_reader = Batch(LEAFTrain(trainer_id), batch_size = config["training"]["batch_size"])

        with logger.training_round() as t:
            print("{} Epoch {} start train".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), epoch_id))
            print(time.time())

            # put your data into the model using `utils.py`

            if count_by_step:
                for step_id, data in enumerate(train_reader()):
                    acc = trainer.run(feeder.feed(data), fetch=["accuracy_0.tmp_0"])
                    step += 1
                    count += 1
                    if count % trainer._step == 0:
                        break
            else:
                trainer.run_with_epoch(
                    train_reader, feeder, fetch=[job._target_names[0], job._target_names[1]], num_epoch=config["training"]["local_epochs"], logger=logger, model_size=model_size)
                    
            if trainer_id == 0:
                save_dir = (output_folder + "/epoch_%d") % epoch_id
                trainer.save_inference_program(output_folder)

logger.end()

time.sleep(3)

if trainer_id == 0:
    beg_time = time.time()
    my_metric = train_test(
                train_test_program=test_program,
                train_test_reader=test_reader,
                train_test_feed=feeder)
    total_evaluation_time = time.time() - beg_time

    if config["dataset"] in regression_dataset:
        metric_name = 'mse'
    elif config["dataset"] in AUC:
        metric_name = 'auc'
    else:
        metric_name = 'accuracy'
    trainer.end_training('{}_{}_{}'.format(my_metric, 0.0, metric_name))
