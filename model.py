import keras

class feGAN():
    def __init__(self, data_dir='./data'):
        self.img_shape = (64, 64, 3)
        self.d_out_shape = (64 / 2**4, 64 / 2**4, 1)

        optimizer = keras.optimizers.Adam(0.0001, 0.5)

        self.d_A = self.__discriminator__()
        self.d_B = self.__discriminator__()

        # disriminator A and disriminator B
        self.d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Generator A->B and Generator B->A
        self.g_AB = self.__generator__()
        self.g_BA = self.__generator__()

        # input A, B
        input_A = Input(shape=self.img_shape)
        input_B = Input(shape=self.img_shape)

        # generate A from A->B and B from B->A from original input (identity)
        selfgen_A = self.g_BA(input_A)
        selfgen_B = self.g_AB(input_B)

        # generate fake A from B->A, B from A->B
        gen_A = self.g_BA(input_B) 
        gen_B = self.g_AB(input_A)

        # regenerate A, B from fake gen_A and gen_B
        regen_A = self.g_BA(gen_B)
        regen_B = self.g_AB(gen_A)

        # check if fake images are 'real'
        class_gen_A = self.d_A(gen_A)
        class_gen_B = self.d_B(gen_B)

        # build model with all components
        self.GAN = Model(
                inputs=[img_A, img_B],
                outputs=[class_gen_A, class_gen_B, regen_A, regen_B, selfgen_A, selfgen_B])
        self.GAN.compile(
                loss=['mse' for i in range(6)],
                loss_weights=[1,1,10,10,1,1],
                optimizer=optimizer)
        


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
        FILTERS = 32
        conv1 = conv2d(init_layer, FILTERS)
        conv2 = conv2d(conv1, FILTERS*2)
        conv3 = conv2d(conv2, FILTERS*4)

        # deconv layers
        dconv1 = upsamp2d(conv3, conv2, FILTERS*2)
        dconv2 = upsamp2d(dconv1, conv1, FILTERS)
        dconv3 = UpSampling2D(size=2)(dconv2)

        # generated image
        img_gen = Conv2D(self.img_shape[2], strides=1, padding='same', activation='tanh')(dconv3)

        return Model(init_layer, img_gen)

    def __discriminator__(self):
        # input layer
        FILTERS = 64 
        init_layer = Input(shape=self.img_shape)

        conv1 = Conv2D(FILTERS, strides=2, padding='same')(init_layer)
        conv1 = ReLU()(conv1)

        conv2 = Conv2D(FILTERS*2, strides=2, padding='same')(conv1)
        conv2 = ReLU()(conv2)
        conv2 = InstanceNormalization()(conv2)

        conv3 = Conv2D(FILTERS*4, strides=2, padding='same')(conv2)
        conv3 = ReLU()(conv3)
        conv3 = InstanceNormalization()(conv3)

        out_layer = Conv2D(1, strides=1, padding='same')(conv3)

        return Model(init_layer, out_layer)
