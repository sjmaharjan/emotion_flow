from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import activations
from keras import initializers

__all__=['MLPAttention']



class MLPAttention(Layer):
    """
    Keras layer to compute an attention vector on an incoming matrix
    and a user provided context vector.

    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)
    # Usage
        enc = Bidirectional(GRU(EMBED_SIZE,return_sequences=True))(...)
        # with user supplied vector
        genre =
        att = MLPAttention()(enc)

    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='ones',
                 v_initializer='glorot_uniform',
                 #Wg_initializer='glorot_uniform',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.v_initializer = initializers.get(v_initializer)
       # self.Wg_initializer = initializers.get(Wg_initializer)
        self.supports_masking = True
        super(MLPAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        #assert type(input_shape) is list and len(input_shape) == 2
        # W: (EMBED_SIZE, units)

        # b: (units,)
        # v: (units,)

        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.units),
                                 initializer=self.kernel_initializer,
                                 trainable=True)

        # self.Wg = self.add_weight(name="W_g{:s}".format(self.name),
        #                           shape=(input_shape[1][-1], self.units),
        #                           initializer=self.Wg_initializer,
        #                           trainable=True)

        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(self.units,),
                                 initializer=self.bias_initializer,
                                 trainable=True)

        self.v = self.add_weight(name="v_{:s}".format(self.name),
                                 shape=(self.units,),
                                 initializer=self.v_initializer,
                                 trainable=True)

        super(MLPAttention, self).build(input_shape)

    def call(self, xs, mask=None):
        # input: [x, u]
        # x: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # g: (BATCH_SIZE, 1,GENRE_EMB_SIZE)

        x= xs

        # print(x.eval())
        # print("x shape {}".format(x.shape.eval()))
        # print(g.eval())
        # print("g shape {}".format(g.shape.eval()))

        # mlp_input = K.dot(x, self.W)
        # mlp_g = K.expand_dims(K.dot(g, self.Wg), axis=1)
        # print('shape MLP:',mlp_input.shape.eval())
        # print('shape_genre',mlp_g.shape.eval())
        #
        # print(mlp_input.eval())
        # print(mlp_g.eval())
        # # b = K.transpose(K.expand_dims(self.b, axis=-1))
        # b=self.b
        # print("b1:", b.shape.eval())
        # print (b.eval())
        # et1 = self.activation(mlp_input + mlp_g + b)  # + )
        # print("et1", et1.shape.eval())
        # print (et1.eval())
        # print("v: ", self.v.shape.eval())
        # # et: (BATCH_SIZE, MAX_TIMESTEPS)
        # et = K.dot(et1, self.v)
        # # at: (BATCH_SIZE, MAX_TIMESTEPS)
        # print("et:", et.shape.eval())
        # print(et.eval())


        # atten_g = K.expand_dims(K.dot(g, self.Wg), axis=1)
        # g=K.squeeze(g,axis=1)
        # atten_g = K.expand_dims(K.dot(g, self.Wg), axis=1)
        # atten_g =K.dot(g, self.Wg)
        # print("atten g  shape:", atten_g.shape.eval())
        # print ('atten g {}'.format(atten_g.eval()))
        et = self.activation(K.dot(x, self.W) + self.b)
        # print("Before dot et:", et.shape.eval())
        et =  K.dot(et, self.v)
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        # et = K.squeeze(et, axis=-1)
        # print("After dot et:", et.shape.eval())

        at = K.softmax(et)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        # print(at.shape.eval())
        # print (at.eval())
        if mask is not None and mask[0] is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        # print(ot.eval())
        return K.sum(ot, axis=1)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(MLPAttention, self).get_config()

