import numpy as np
import mxnet as mx
import cv2


def get_model(ctx, input_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, input_size[0], input_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, args):
        self.args = args
        ctx = mx.cpu()
        self.input_size = [112, 112]
        self.model = get_model(ctx, self.input_size, args.model, 'fc1')

    def get_feature(self, img):
        img = cv2.resize(img, (self.input_size[0], self.input_size[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        im = np.expand_dims(img, axis=0)
        data = mx.nd.array(im)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        return embedding
