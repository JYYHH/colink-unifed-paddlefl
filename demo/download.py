import flbenchmark.datasets
import json

config = json.load(open('config.json', 'r'))

flbd = flbenchmark.datasets.FLBDatasets('../data')

print("Downloading Data...")

dataset_name = (
                'student_horizontal',
                'breast_horizontal',
                'default_credit_horizontal',
                'give_credit_horizontal',
                'vehicle_scale_horizontal'
                )



for x in dataset_name:
    if config["dataset"] == x:
        # print(x)
        train_dataset, test_dataset = flbd.fateDatasets(x)
        # print(train_dataset)
        flbenchmark.datasets.convert_to_csv(train_dataset, out_dir='../csv_data/{}_train'.format(x))
        # print(x)
        if x != 'vehicle_scale_horizontal':
            flbenchmark.datasets.convert_to_csv(test_dataset, out_dir='../csv_data/{}_test'.format(x))

leaf = (
    'femnist',
    'reddit'
)

for x in leaf:
    if config["dataset"] == x:
        my_dataset = flbd.leafDatasets(x)