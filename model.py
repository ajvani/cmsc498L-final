import os
import keras
import tensorflow as tf
import numpy as np

from data import CelebDataLoader

from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D

class o2mCycleGAN():
    def __init__(self, atts=None, data_dir='./data'):
        default_atts = ['Bald', 'Mustache', 'Eyeglasses', 'Bangs']
        if atts == None:
            self.atts = default_atts
        else:
            self.atts = atts

        self.n_atts = len(self.atts)
        self.data_dir = data_dir
        self.img_shape = (64, 64, 3)

        d_s = int(64 / 2**4)
        self.d_out_shape = (d_s, d_s, 1)

        optimizer = keras.optimizers.Adam(0.0001, 0.5)

        # truth -> fake discriminators
        self.d_T_lst = [self.__discriminator__() for _ in self.atts]

        # fake -> truth discriminators
        self.d_F_lst = [self.__discriminator__() for _ in self.atts]

        # disriminator A and disriminator B
        for (disc_TF, disc_FT) in zip(self.d_T_lst, self.d_F_lst):
            disc_TF.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
            disc_FT.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # build generator truth->fake and generators fake->truth
        self.g_TF_lst = [self.__generator__() for _ in self.atts]
        self.g_FT_lst = [self.__generator__() for _ in self.atts]

        # ground truth input + fake input for each network
        self.input_T_lst = [Input(shape=self.img_shape) for _ in self.atts]
        self.input_F_lst = [Input(shape=self.img_shape) for _ in self.atts]

        # generate A from A->B and B from B->A from original input (identity)
        selfgen_T_lst = [g_FT(input_T) for (input_T, g_FT) in zip(self.input_T_lst, self.g_FT_lst)]
        selfgen_F_lst = [g_TF(input_F) for (input_F, g_TF) in zip(self.input_F_lst, self.g_TF_lst)]

        # generate fake A from B->A, B from A->B
        # fake generations
        gen_T_lst = [g_FT(input_F) for (input_F, g_FT) in zip(self.input_F_lst, self.g_FT_lst)]
        gen_F_lst = [g_TF(input_T) for (input_T, g_TF) in zip(self.input_T_lst, self.g_TF_lst)]

        # regenerate A, B from fake gen_A and gen_B
        regen_T_lst = [g_TF(gen_F) for (g_TF, gen_F) in zip(self.g_TF_lst, gen_F_lst)]
        regen_F_lst = [g_FT(gen_T) for (g_FT, gen_T) in zip(self.g_FT_lst, gen_T_lst)]

        # check if fake images are 'real'
        class_gen_T_lst = [d_T(gen_T) for (d_T, gen_T) in zip(self.d_T_lst, gen_T_lst)]
        class_gen_F_lst = [d_F(gen_F) for (d_F, gen_F) in zip(self.d_F_lst, gen_F_lst)]

        # build model with all components
        self.GANs = [ 
                Model(
                inputs=[img_A, img_B],
                outputs=[class_gen_A, class_gen_B, regen_A, regen_B, selfgen_A, selfgen_B])
                for (img_A, img_B, class_gen_A, class_gen_B, regen_A, regen_B, selfgen_A, selfgen_B)
                in zip(self.input_T_lst, self.input_F_lst, class_gen_T_lst, class_gen_F_lst,
                    regen_T_lst, regen_F_lst, selfgen_T_lst, selfgen_F_lst)]

        for GAN in self.GANs:
            GAN.compile(
                loss=['mse' for i in range(6)],
                loss_weights=[1,1,10,10,1,1],
                optimizer=optimizer)
    


    # returns a new generator for the cycleGAN
    def __generator__(self):  
        def conv2d(in_layer, filters):
            conv = Conv2D(filters, kernel_size=4, strides=2, padding='same')(in_layer)
            conv = ReLU()(conv)
            conv = BatchNormalization(axis=3)(conv)
            return conv

        def upsamp2D(in_layer, conc_layer, filters):
            upsamp = UpSampling2D(size=2)(in_layer)
            upsamp = BatchNormalization(axis=3)(upsamp)
            upsamp = Dropout(0.1)(upsamp)
            upsamp = Concatenate()([upsamp, conc_layer])
            return upsamp

        # input layer
        init_layer = Input(shape=self.img_shape)

        # conv 2d layers
        FILTERS = 16
        conv1 = conv2d(init_layer, FILTERS)
        conv2 = conv2d(conv1, FILTERS*2)
        conv3 = conv2d(conv2, FILTERS*4)
        conv4 = conv2d(conv3, FILTERS*8)

        # deconv layers
        dconv1 = upsamp2D(conv4, conv3, FILTERS*4)
        dconv2 = upsamp2D(dconv1, conv2, FILTERS*2)
        dconv3 = upsamp2D(dconv2, conv1, FILTERS)
        dconv4 = UpSampling2D(size=2)(dconv3)

        # generated image
        img_gen = Conv2D(self.img_shape[2], kernel_size=4, strides=1, padding='same', activation='tanh')(dconv4)

        return Model(init_layer, img_gen)

    # discriminator: checks if input is part of class <att> or not
    def __discriminator__(self):
        # input layer
        FILTERS = 32 
        init_layer = Input(shape=self.img_shape)

        conv1 = Conv2D(FILTERS, kernel_size=4, strides=2, padding='same')(init_layer)
        conv1 = ReLU()(conv1)

        conv2 = Conv2D(FILTERS*2, kernel_size=4, strides=2, padding='same')(conv1)
        conv2 = ReLU()(conv2)
        conv2 = BatchNormalization(axis=3)(conv2)

        conv3 = Conv2D(FILTERS*4, kernel_size=4, strides=2, padding='same')(conv2)
        conv3 = ReLU()(conv3)
        conv3 = BatchNormalization(axis=3)(conv3)

        out_layer = Conv2D(1, kernel_size=4, strides=1, padding='same')(conv3)

        return Model(init_layer, out_layer)

    def build_data(self, method='train'):
        data_loader = CelebDataLoader(self.data_dir, self.atts, method=method)
        data_it = data_loader.__iterator__()
        _img, _label = data_it.get_next()
        session = tf.Session()
        session.run(data_it.initializer)

        data = {}
        for label_i in range(self.n_atts):
            data[label_i] = []

        while True:
            try:
                img, label = _img.eval(session=session), _label.eval(session=session)
                _img, _label = data_it.get_next()
            except tf.errors.OutOfRangeError:
                break

            for (i,cls) in enumerate(label):
                if cls == 1:
                    data[i].append(img)
        return data
        
        

    def train(self, epochs=50, batch_size=1):
        data = self.build_data()

        valid = np.ones((batch_size,) + self.d_out_shape)
        fake = np.zeros((batch_size,) + self.d_out_shape)
        for epoch in range(epochs):
            for label in data.keys():
                d_T = self.d_T_lst[label]
                d_F = self.d_F_lst[label]
                g_TF = self.g_TF_lst[label]
                g_FT = self.g_FT_lst[label]

                corr_att = self.atts[label]

                # for label i, generate fake for all labels != i
                for img_A in data[label]:
                    for fake_label in range(self.n_atts):
                        if fake_label == label: continue
                        for img_B in data[fake_label]:
                            img_A = data[label]

                            fake_F = g_TF.predict(img_A)
                            fake_T = g_FT.predict(img_B)

                            dT_loss_real = d_T.tain_on_batch(img_A, valid)
                            dT_loss_fake = d_T.train_on_batch(fake_T, fake)
                            dT_loss = 0.5 * np.add(dT_loss_real, dT_loss_fake)

                            dF_loss_real = d_F.train_on_batch(img_B, valid)
                            dF_loss_fake = d_T.train_on_batch(fake_F, fake)
                            dF_loss = 0.5 * np.add(dF_loss_real, dF_loss_fake)

                            d_loss = 0.5 * np.add(dT_loss, dF_loss)

                            # Train Gens
                            g_loss = self.GANs[label].train_on_batch([img_A, img_B], [valid, valid, img_A, img_B, img_A, img_B])

                print ("[Epoch %d/%d] [label: %d]" % (epoch, epochs, label))
                # print ("[Epoch %d/%d] [label: %d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f]" \
                #                                                         % ( epoch, epochs, label,
                #                                                             d_loss[0], 100*d_loss[1],
                #                                                             g_loss[0],
                #                                                             np.mean(g_loss[1:3]),
                #                                                             np.mean(g_loss[3:5]),
                #                                                             np.mean(g_loss[5:6])))

        print("\n\n")


            


