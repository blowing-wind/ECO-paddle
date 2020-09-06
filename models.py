import paddle.fluid as fluid
from model_zoo.model_load import ECOfull
import numpy as np
from paddle.fluid.dygraph import Linear, Dropout

class TSN(fluid.dygraph.Layer):
    def __init__(self, num_class, num_segments, modality,
                 base_model='ECOfull', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.base_model_name = base_model
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_base_model(self, base_model):
        if base_model == 'ECO':
            raise NotImplementedError

        elif base_model == 'ECOfull' :
            self.base_model = ECOfull(num_segments=self.num_segments)
            self.base_model.last_layer_name = 'fc_final'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
        else:
            raise ValueError('Not supported arch: {}!'.format(base_model))
    
    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).weight.shape[0]
        setattr(self.base_model, self.base_model.last_layer_name, Dropout(p=self.dropout))
        self.new_fc = Linear(feature_dim, num_class, act='softmax', 
        param_attr=fluid.ParamAttr(learning_rate=1.0), bias_attr=fluid.ParamAttr(learning_rate=2.0))
    
    def forward(self, input, label=None):
        input_var = fluid.layers.reshape(input, [-1, 3] + input.shape[-2:])
        base_out = self.base_model(input_var)
        y = self.new_fc(base_out)
        
        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y
    
    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224


if __name__ == '__main__':
    with fluid.dygraph.guard():
        network = TSN(101, 16, 'RGB', 'ECOfull')
        img = np.zeros([2, 16, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        outs = network(img).numpy()
        print(outs.shape)
        print(outs)
    