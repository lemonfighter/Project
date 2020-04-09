import numpy as np
import pickle


def softplus(a, d=False) -> np.array:
    def normal():
        return np.log(1 + np.exp(a))

    def d_():
        return 1 / (1 + np.exp(-a))
    if d:
        return d_()
    else:
        return normal()


def relu(a, d=False) -> np.array:
    if d:
        return [1 if x >= 0 else 0 for x in a]
    else:
        return np.maximum(a, 0)


def tanh(a, d=False) -> np.array:
    def normal():
        return 2/(1+np.exp(-2*a)) - 1

    def d_():
        return 1-np.square(normal())
    if d:
        return d_()
    else:
        return normal()


def liner(a, d=False) -> np.array:
    if d:
        return np.zeros(a.shape) + 1
    else:
        return a


def sigmoid(a, d=False) -> np.array:
    def normal():
        return 1.0 / (1.0 + np.exp(-a))

    def d_():
        return normal()*(np.array(1)-normal())
    if d:
        return d_()
    else:
        return normal()


def z_score(data):
    data_mean = np.mean(data)
    data_std = np.std(data)
    return (data - data_mean) / data_std


def max_min(data, maximun=1., negative=False):
    data_max = np.amax(data)
    data_min = np.amin(data)
    newdata = (data - data_min) / ((data_max - data_min)+1e-5) * maximun
    if negative:
        return (newdata - maximun/2) * 2
    else:
        return newdata


def fix_max_min(data, max_values: float, min_values: float):
    return (data - min_values) / (max_values - min_values)


def get_pop(low, high, number_of_x: int, layermap: list) -> list:
    pop = [(np.random.uniform(low, high, size=(number_of_x, layermap[0])), np.random.uniform(low, high, size=layermap[0]))]
    for index in range(1, len(layermap)):
        pop.append((np.random.uniform(low, high, size=(layermap[index-1], layermap[index])), np.random.uniform(low, high, size=layermap[index])))
    return pop


class NeuralNetwork:
    def __init__(self, layermap=None, activtion_function_map=None):
        self.LayerMap = layermap
        self.Pop = None
        self.ActivtionFunctionMap = activtion_function_map

    def load_pop(self, pop_path):
        with open(pop_path, 'rb') as f:
            savedata = pickle.load(f)
        self.LayerMap = savedata[0]
        self.Pop = savedata[1]
        self.ActivtionFunctionMap = savedata[2]

    def save_pop(self, file_name):
        save_data = [self.LayerMap, self.Pop, self.ActivtionFunctionMap]
        with open(file_name, 'wb') as f:
            pickle.dump(save_data, f)

    def creat_pop(self, low, high):
        w_list = []
        b_list = []
        for index in range(1, len(self.LayerMap)):
            w_list.append(np.random.uniform(low, high, size=(self.LayerMap[index - 1], self.LayerMap[index])))
            b_list.append(np.random.uniform(low, high, size=self.LayerMap[index]))
        self.Pop = [w_list, b_list]

    def get_result(self, npdata: list) -> (list, list):
        def layermarker():
            anp = np.dot(data_input, ws) + bs
            return af(anp), anp
        if isinstance(npdata, list):
            npdata = np.array(npdata)
        data_input = npdata
        layer_output_list = [[np.array(npdata), 0]]
        for ws, bs, af in zip(self.Pop[0], self.Pop[1], self.ActivtionFunctionMap):
            data_input, un_af = layermarker()
            layer_output_list.append([np.array(data_input), np.array(un_af)])
        return np.array(data_input), layer_output_list

    def train(self, training_datas, label_datas, learning_rate):
        if len(training_datas) != len(label_datas):
            raise ValueError("size of the training_datas and label_datas is different")
        z_w = [np.zeros(w.shape) for w in self.Pop[0]]
        z_b = [np.zeros(b.shape) for b in self.Pop[1]]
        batch_size = len(training_datas)
        for x, y in zip(training_datas, label_datas):
            output, layer_output = self.get_result(x)
            d_w, d_b = self.get_gradient_change(layer_output, y)
            z_w = [zw+dw for zw, dw in zip(z_w, d_w)]
            z_b = [zb+db for zb, db in zip(z_b, d_b)]
        self.Pop[0] = [w-learning_rate*nw/batch_size for w, nw in zip(self.Pop[0], z_w)]
        self.Pop[1] = [b-learning_rate*nb/batch_size for b, nb in zip(self.Pop[1], z_b)]

    def get_gradient_change(self, layer_output, y):
        z_w = [np.zeros(w.shape) for w in self.Pop[0]]
        z_b = [np.zeros(b.shape) for b in self.Pop[1]]
        a, z = layer_output[-1]
        af = self.ActivtionFunctionMap[-1]
        if af is None:
            d = self.cot_derivative(a, y)
        else:
            d = self.cot_derivative(a, y) * af(z, d=True)
        z_b[-1] += d
        a, z = layer_output[-2]
        z_w[-1] += np.dot(a.reshape(-1, 1), d).reshape(-1, len(d))
        for index in range(2, len(self.LayerMap)):
            a, z = layer_output[-index]
            af = self.ActivtionFunctionMap[-index]
            w = self.Pop[0][-index+1]
            if af is None:
                d = np.dot(d, w.transpose())
            else:
                d = np.dot(d, w.transpose()) * af(z, d=True)
            z_b[-index] += d
            a, z = layer_output[-index-1]
            z_w[-index] += np.dot(a.reshape(-1, 1), d.reshape(1, len(d)))
        return z_w, z_b

    def cot_derivative(self, xs, ys):
        return (xs-ys) * 2


