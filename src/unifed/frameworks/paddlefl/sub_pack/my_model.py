import paddle.fluid as fluid
import json

config = json.load(open('config.json', 'r'))

class Logistic_Regression(object):
    def __init__(self, in_dim, out_dim, *_):
        self.dim = [in_dim, out_dim]

    def my_network(self):
        self.inputs = fluid.layers.data(
            name='vector', shape=[self.dim[0]], dtype="float32")
        self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        self.predict = fluid.layers.fc(name = 'predict',
                                       input=self.inputs,
                                       size=self.dim[-1],
                                    #    param_attr=fluid.initializer.ConstantInitializer(value=0.0),
                                       act='sigmoid')
        self.sum_cost = fluid.layers.softmax_with_cross_entropy(
            logits=self.predict, label=self.label)
        self.accuracy = fluid.layers.accuracy(
            input=self.predict, label=self.label)
        self.loss = fluid.layers.mean(self.sum_cost)
        self.startup_program = fluid.default_startup_program()
        return (self.dim[0] + 1) * self.dim[-1] * 4.0

class Linear_Regression(object):
    def __init__(self, in_dim, out_dim, *_):
        self.dim = [in_dim]

    def my_network(self):
        self.inputs = fluid.layers.data(
            name='vector', shape=[self.dim[0]], dtype="float32")
        self.label = fluid.layers.data(name='label', shape=[1], dtype='float32')
        self.predict = fluid.layers.fc(input=self.inputs,
                                       size=1,
                                    #    param_attr=fluid.initializer.ConstantInitializer(value=0.0),
                                       act=None)
        self.loss = fluid.layers.mse_loss(
            input=self.predict, label=self.label)
        self.startup_program = fluid.default_startup_program()
        return (self.dim[0] + 1) * 1 * 4.0

class MLP(object):
    def __init__(self, in_dim, out_dim, hidden):
        self.dim = [in_dim] + hidden + [out_dim]
        self.hidden = hidden

    def my_network(self):
        if len(self.hidden) == 1:
            self.inputs = fluid.layers.data(
                name='vector', shape=[self.dim[0]], dtype="float32")
            self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            self.mid = fluid.layers.fc(input=self.inputs,
                                    size=self.dim[1],
                                    act='relu')
            self.predict = fluid.layers.fc(input=self.mid,
                                        size=self.dim[-1],
                                        act='softmax')
            self.sum_cost = fluid.layers.cross_entropy(
                input=self.predict, label=self.label)
            self.accuracy = fluid.layers.accuracy(
                input=self.predict, label=self.label)
            self.loss = fluid.layers.mean(self.sum_cost)
            self.startup_program = fluid.default_startup_program()
            return ( ((self.dim[0] + 1) * self.dim[1]) + ((self.dim[1] + 1) * self.dim[-1]) ) * 4.0
        
        elif len(self.hidden) == 2:
            self.inputs = fluid.layers.data(
                name='vector', shape=[self.dim[0]], dtype="float32")
            self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            self.mid1 = fluid.layers.fc(input=self.inputs,
                                    size=self.dim[1],
                                    act='relu')
            self.mid2 = fluid.layers.fc(input=self.mid1,
                                    size=self.dim[2],
                                    act='relu')                   
            self.predict = fluid.layers.fc(input=self.mid2,
                                        size=self.dim[-1],
                                        act='softmax')
            self.sum_cost = fluid.layers.cross_entropy(
                input=self.predict, label=self.label)
            self.accuracy = fluid.layers.accuracy(
                input=self.predict, label=self.label)
            self.loss = fluid.layers.mean(self.sum_cost)
            self.startup_program = fluid.default_startup_program()
            return ( ((self.dim[0] + 1) * self.dim[1]) + ((self.dim[1] + 1) * self.dim[2]) + ((self.dim[2] + 1) * self.dim[-1]) ) * 4.0          
        
        elif len(self.hidden) == 3:
            self.inputs = fluid.layers.data(
                name='vector', shape=[self.dim[0]], dtype="float32")
            self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            self.mid1 = fluid.layers.fc(input=self.inputs,
                                    size=self.dim[1],
                                    act='relu')
            self.mid2 = fluid.layers.fc(input=self.mid1,
                                    size=self.dim[2],
                                    act='relu')     
            self.mid3 = fluid.layers.fc(input=self.mid2,
                                    size=self.dim[3],
                                    act='relu')               
            self.predict = fluid.layers.fc(input=self.mid3,
                                        size=self.dim[-1],
                                        act='softmax')
            self.sum_cost = fluid.layers.cross_entropy(
                input=self.predict, label=self.label)
            self.accuracy = fluid.layers.accuracy(
                input=self.predict, label=self.label)
            self.loss = fluid.layers.mean(self.sum_cost)
            self.startup_program = fluid.default_startup_program()
            return ( ((self.dim[0] + 1) * self.dim[1]) + ((self.dim[1] + 1) * self.dim[2]) + ((self.dim[2] + 1) * self.dim[3]) + ((self.dim[3] + 1) * self.dim[-1]) ) * 4.0  

class CNN(object):
    def __init__(self, *_):
        pass

    def my_network(self):
        self.inputs = fluid.layers.data(
            name='img', shape=[1, 28, 28], dtype="float32")
        self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        self.conv_pool_1 = fluid.nets.simple_img_conv_pool(
            input=self.inputs,
            num_filters=20,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act='relu')
        self.conv_pool_2 = fluid.nets.simple_img_conv_pool(
            input=self.conv_pool_1,
            num_filters=50,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act='relu')

        self.predict = self.predict = fluid.layers.fc(input=self.conv_pool_2,
                                                      size=62,
                                                      act='softmax')
        self.cost = fluid.layers.cross_entropy(
            input=self.predict, label=self.label)
        self.accuracy = fluid.layers.accuracy(
            input=self.predict, label=self.label)
        self.loss = fluid.layers.mean(self.cost)
        self.startup_program = fluid.default_startup_program()
        return 0.0


class lenet(object):
    def __init__(self, intput_dim, output_dim, *_):
        self.indim = intput_dim
        self.outdim = output_dim

    def my_network(self):
        self.inputs = fluid.layers.data(
                name='vector', shape=[1, 32, 32], dtype="float32")

        self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        self.conv_pool_1 = fluid.nets.simple_img_conv_pool(
            input=self.inputs,
            num_filters=6,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act='relu')

        self.conv_pool_2 = fluid.nets.simple_img_conv_pool(
            input=self.conv_pool_1,
            num_filters=16,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act='relu')

        self.fc1 = fluid.layers.fc(input=self.conv_pool_2,
                                                      size=120,
                                                      act='relu')
        self.fc2 = fluid.layers.fc(input=self.fc1,
                                                      size=84,
                                                      act='relu')                                              
        self.predict = fluid.layers.fc(input=self.fc2,
                                                      size=self.outdim,
                                                      act='softmax')
        self.cost = fluid.layers.cross_entropy(
            input=self.predict, label=self.label)
        self.accuracy = fluid.layers.accuracy(
            input=self.predict, label=self.label)
        self.loss = fluid.layers.mean(self.cost)
        self.startup_program = fluid.default_startup_program()
        return 264504.0

class lstm(object):
    def __init__(self, output_dim, embedding_size, hidden_size):
        self.outdim = output_dim
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
    def my_network(self):
        self.inputs = fluid.layers.data(
                name='vector', shape=[10, 1], dtype="int64")
        self.label = fluid.layers.data(name='label', shape=[10, 1], dtype='int64')
        self.emb = fluid.layers.embedding(input=self.inputs, 
                                          size=[self.outdim, self.embedding_size], 
                                          is_sparse=True)

        print(self.emb)


        self.lstm = fluid.layers.rnn(cell = fluid.layers.LSTMCell(
                                                            hidden_size=self.hidden_size), 
                                     inputs = self.emb,
                                    )

        print(self.lstm)

        # self.fc1 = fluid.layers.fc(input=self.lstm[0],
        #                            size=self.embedding_size,
        #                            num_flatten_dims=2)

        # print(self.fc1)

        self.predict = fluid.layers.fc(input=self.lstm[0],
                                   size=self.outdim,
                                   num_flatten_dims=2)

        print(self.predict)

        self.sum_cost = fluid.layers.softmax_with_cross_entropy(
            logits=self.predict, label=self.label, ignore_index = 0, axis = 2)

        print(self.sum_cost)

        self.loss = fluid.layers.mean(self.sum_cost)

        print(self.loss)
        
        self.startup_program = fluid.default_startup_program()

        return 33979748.0

def get_model(model_name, *param):
    if model_name == "logistic_regression":
        return Logistic_Regression(*param)
    elif model_name == "linear_regression":
        return Linear_Regression(*param)
    elif model_name[:3] == "mlp":
        return MLP(*param)
    elif model_name == "cnn":
        return CNN(*param)
    elif model_name == "lenet":
        return lenet(*param)
    else:
        assert 1==0