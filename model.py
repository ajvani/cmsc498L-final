import keras

class feGAN():
    def __init__(self, data_dir='./data', atts=None):
        self.img_shape = (64, 64, 3)

        self.target_atts = ['Bald', 'Bangs', 'Black_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Mustache', 'Young']
        if atts:
            self.target_atts = atts

    # returns a new generator for the cycleGAN
    def __generator__(self):  
        def conv2d(in_layer, filters):
            conv = Conv2D(filters, strides=2, padding='same')(in_layer)
            conv = ReLU()(conv)
            conv = InstanceNormalization()(conv)
            return conv

        def upsamp2D(in_layer, conc_layer, filters):
            upsamp = UpSampling2D(size=2)(in_layer)
            upsamp = BatchNormalization(mode=2, axis=3)(upsamp)
            upsamp = Dropout(0.1)(upsamp)
            upsamp = Concatenate()([upsamp, conc_layer])
            return upsamp

        # input layer
        init_layer = Input(shape=self.img_shape)

        # conv 2d layers
        filters = 32
        conv1 = conv2d(init_layer, filters)
        conv2 = conv2d(conv1, filters*2)
        conv3 = conv2d(conv2, filters*4)

        # deconv layers
        dconv1 = upsamp2d(conv3, conv2, filters*2)
        dconv2 = upsamp2d(dconv1, conv1, filters)
        dconv3 = UpSampling2D(size=2)(dconv2)

        # generated image
        img_gen = Conv2D(self.img_shape[2], strides=1, padding='same', activation='tanh')(dconv3)

        return Model(init_layer, img_gen)

    def __discriminator__(self):
        # input layer
        init_layer = Input(shape=self.img_shape)



