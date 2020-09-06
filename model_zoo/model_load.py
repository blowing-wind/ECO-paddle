import paddle.fluid as fluid
from .layer_factory import get_basic_layer, parse_expr
import yaml


class ECOfull(fluid.dygraph.Layer):
    def __init__(self, model_path='model_zoo/ECOfull.yaml', num_segments=4):

        super(ECOfull, self).__init__()

        self.num_segments = num_segments

        manifest = yaml.load(open(model_path), Loader=yaml.FullLoader)

        layers = manifest['layers']

        self._channel_dict = dict()

        self._op_list = list()
        for l in layers:
            out_var, op, in_var = parse_expr(l['expr'])
            if op != 'Concat' and op != 'Eltwise':
                id, out_name, module, out_channel, in_name = get_basic_layer(l,
                                                                3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]],
                                                                             conv_bias=True if op == 'Conv3d' else True, num_segments=num_segments)

                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            elif op == 'Concat':
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel
            else:
                self._op_list.append((id, op, out_var[0], in_var))
                channel = self._channel_dict[in_var[0]]
                self._channel_dict[out_var[0]] = channel


    def forward(self, input):
        data_dict = dict()
        data_dict[self._op_list[0][-1]] = input

        for op in self._op_list:
            if op[1] != 'Concat' and op[1] != 'InnerProduct' and op[1] != 'Eltwise' and op[1] != 'ReLU' and op[1] != 'Pooling3d':
                # first 3d conv layer judge, the last 2d conv layer's output must be transpose from 4d to 5d
                if op[0] == 'res3a_2':
                    layer_output = data_dict[op[-1]]
                    layer_output = fluid.layers.reshape(layer_output, [-1, self.num_segments] + layer_output.shape[1:])
                    layer_transpose_output = fluid.layers.transpose(layer_output, perm=[0, 2, 1, 3, 4])
                    data_dict[op[2]] = getattr(self, op[0])(layer_transpose_output)
                else:
                    data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
            
            elif op[1] == 'ReLU':
                data_dict[op[2]] = fluid.layers.relu(data_dict[op[-1]])
            elif op[1] == 'Pooling3d':
                if op[0] == 'global_pool2D_reshape_consensus':
                    layer_output = data_dict[op[-1]]
                    layer_output = fluid.layers.reshape(layer_output, [-1, self.num_segments] + layer_output.shape[1:])
                    this_input = fluid.layers.transpose(layer_output, perm=[0, 2, 1, 3, 4])
                else:
                    this_input = data_dict[op[-1]]
                method, ks, stride, padding = getattr(self, op[0])
                if method == 'max':
                    data_dict[op[2]] = fluid.layers.pool3d(this_input, ks, 'max', stride, padding, ceil_mode=True)
                elif method == 'ave':
                    data_dict[op[2]] = fluid.layers.pool3d(this_input, ks, 'avg', stride, padding, ceil_mode=True)
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                x = fluid.layers.reshape(x, [x.shape[0], -1])
                data_dict[op[2]] = getattr(self, op[0])(x)
            elif op[1] == 'Eltwise':
                try:
                    data_dict[op[2]] = fluid.layers.elementwise_add(data_dict[op[-1][0]], data_dict[op[-1][1]])
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].shape)
                    raise
                # x = data_dict[op[-1]]
                # data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = fluid.layers.concat([data_dict[x] for x in op[-1]], 1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].shape)
                    raise
        # print output data size in each layers
        # for k in data_dict.keys():
        #     print(k,": ",data_dict[k].size())
        # exit()

        # "self._op_list[-1][2]" represents: last layer's name(e.g. fc_action)
        return data_dict[self._op_list[-1][2]]
