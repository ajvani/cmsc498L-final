import keras

class o2mCycleGAN():
    def __init__(self, atts, data_dir='./data'):
        self.img_shape = (64, 64, 3)
        self.d_out_shape = (64 / 2**3, 64 / 2**3, 1)

        optimizer = keras.optimizers.Adam(0.0001, 0.5)

        # truth -> fake discriminators
        self.d_TF_lst = [self.__discriminator__() for _ in atts]

        # fake -> truth discriminators
        self.d_FT_lst = [self.__discriminator__() for _ in atts]

        # disriminator A and disriminator B
        for (disc_TF, disc_FT) in zip(self.d_TF_lst, self.d_FT_lst):
            disc_TF.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
            disc_FT.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # build generator truth->fake and generators fake->truth
        self.g_TF_lst = [self.__generator__() for _ in atts]
        self.g_FT_lst = [self.__generator__() for _ in atts]

        # ground truth input + fake input for each network
        self.input_T_lst = [Input(shape=self.img_shape) for _ in atts]
        self.input_F_lst = [Input(shape=self.img_shape) for _ in atts]

        # generate A from A->B and B from B->A from original input (identity)
        selfgen_T_lst = [g_FT(input_T) for (input_T, g_FT) in zip(self.input_T_lst, self.g_FT_lst)]
        selfgen_F_lst = [g_TF(input_F) for (input_F, g_FT) in zip(self.input_F_lst, self.g_TF_lst)]

        # generate fake A from B->A, B from A->B
        # fake generations
        gen_T_lst = [g_FT(input_F) for (input_F, g_FT) in zip(self.input_F_lst, self.g_FT_lst)]
        gen_F_lst = [g_TF(input_T) for (input_T, g_TF) in zip(self.input_T_lst, self.g_TF_lst)]

        # regenerate A, B from fake gen_A and gen_B
        regen_T_lst = [g_TF(gen_F) for (g_TF, gen_F) in zip(self.g_TF_lst, gen_F_lst)]
        regen_F_lst = [g_FT(gen_T) for (g_FT, gen_T) in zip(self.g_FT_lst, gen_T_lst)]

        # check if fake images are 'real'
        class_gen_T_lst = [d_T(gen_T) for (d_T, gen_T) in zip(self.d_TF_lst, gen_T_lst)]
        class_gen_F_lst = [d_F(gen_T) for (d_F, gen_F) in zip(self.d_FT_lst, gen_F_lst)]

        # build model with all components
        self.GANs = [ 
                Model(
                inputs=[img_A, img_B],
                outputs=[class_gen_A, class_gen_B, regen_A, regen_B, selfgen_A, selfgen_B])
                for (img_A, img_B, class_gen_A, class_gen_B, regen_A, regen_B, selfgen_A, selfgen_B)
                in zip(self.input_T_lst, self.input_F_lst, class_gen_T_lst, class_gen_F_lst,
                    regen_T_lst, regen_F_lst, selfgen_T_lst, selfgen_F_lst)]
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
