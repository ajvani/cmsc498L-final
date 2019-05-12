import os
import keras
import tensorflow as tf
import numpy as np
import datetime

from data import CelebDataLoader

from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D

import matplotlib.pyplot as plt


class o2mCycleGAN():
    def __init__(self, atts=None, data_dir='./data', model_dir='./model_weights', load_pretrained=False):
        default_atts = ['Bald', 'Mustache', 'Eyeglasses', 'Bangs']
        if atts == None:
            self.atts = default_atts
        else:
            self.atts = atts

        self.model_dir = model_dir
        self.n_atts = len(self.atts)
        self.data_dir = data_dir
        self.img_shape = (64, 64, 3)

        d_s = int(64 / 2**3)
        self.d_out_shape = (d_s, d_s, 1)

        optimizer = keras.optimizers.Adam(0.0001, 0.5)

        print('Setting up Models')
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
        # GAN[i] => generates an image with label[i] = 1 using images with label[i] = 0
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

        print('Models Setup')
        if load_pretrained:
            print('Loading saved weights')
            self.load_weights()
    


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
    
    def load_weights(self):
        print('Loading weights from path')
        model_dir = self.model_dir
        if not os.path.exists(model_dir):
            return

        # save all the discriminators
        for i, d_T in enumerate(self.d_T_lst):
            d_T_path = os.path.join(model_dir, 'dT_'+self.atts[i]+'.h5')
            if os.path.exists(d_T_path):
                d_T.load_weights(d_T_path)
                d_T.trainable = False

        for i, d_T in enumerate(self.d_F_lst):
            d_T_path = os.path.join(model_dir, 'dT_'+self.atts[i]+'.h5')
            if os.path.exists(d_T_path):
                d_T.load_weights(d_T_path)
                d_T.trainable = False

        # save generators
        for i, g_TF in enumerate(self.g_TF_lst):
            g_TF_path = os.path.join(model_dir, 'gTF_'+self.atts[i]+'.h5')
            if os.path.exists(g_TF_path):
                g_TF.load_weights(g_TF_path)
                g_TF.trainable = False

        for i, g_TF in enumerate(self.g_FT_lst):
            g_TF_path = os.path.join(model_dir, 'gTF_'+self.atts[i]+'.h5')
            if os.path.exists(g_TF_path):
                g_TF.load_weights(g_TF_path)
                g_TF.trainable = False

        # save GANs
        for i, gan in enumerate(self.GANs):
            gan_path = os.path.join(model_dir, 'gan_'+self.atts[i]+'.h5')
            if os.path.exists(gan_path):
                gan.load_weights(gan_path)
                gan.trainable = False
        print('Loaded all weights')
        
    
    def save_models(self):
        model_dir = self.model_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        # save all the discriminators
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('%s Storing Discriminators' % curr_time)
        for i, d_T in enumerate(self.d_T_lst):
            d_T_path = os.path.join(model_dir, 'dT_'+self.atts[i]+'.h5')
            d_T.save_weights(d_T_path)

        for i, d_T in enumerate(self.d_F_lst):
            d_T_path = os.path.join(model_dir, 'dF_'+self.atts[i]+'.h5')
            d_T.save_weights(d_T_path)

        # save generators
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('%s Storing Generators' % curr_time)
        for i, g_TF in enumerate(self.g_TF_lst):
            g_TF_path = os.path.join(model_dir, 'gTF_'+self.atts[i]+'.h5')
            g_TF.save_weights(g_TF_path)

        for i, g_TF in enumerate(self.g_FT_lst):
            g_TF_path = os.path.join(model_dir, 'gTF_'+self.atts[i]+'.h5')
            g_TF.save_weights(g_TF_path)

        # save GANs
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('%s Storing GANs' % curr_time)
        for i, gan in enumerate(self.GANs):
            gan_path = os.path.join(model_dir, 'gan_'+self.atts[i]+'.h5')
            gan.save_weights(gan_path)
        
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('%s Finished Storing Models' % curr_time)

    def train(self, epochs=50, batch_size=1):
        # load dataset twice: A = truth, B = fake. Skip if label_A[i] == label_B[i]
        print('Loading Dataset')
        data_loader_A = CelebDataLoader(self.data_dir, self.atts)
        data_loader_B = CelebDataLoader(self.data_dir, self.atts)


        print('Starting Training')
        for epoch in range(1,epochs+1):
            start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print ("%s [Epoch %d/%d]" % (start_time, epoch, epochs))
            for label in range(self.n_atts):
                curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("\t%s [label: %s]" % (curr_time, self.atts[label]))
                valid = np.ones((batch_size,) + self.d_out_shape)
                fake = np.zeros((batch_size,) + self.d_out_shape)

                d_T = self.d_T_lst[label]
                d_F = self.d_F_lst[label]

                g_TF = self.g_TF_lst[label]
                g_FT = self.g_FT_lst[label]

                corr_att = self.atts[label]

                # for label i, generate fake for all labels != i

                for i, (img_A, label_A) in enumerate(data_loader_A):
                    if label_A[label] == 1: continue
                    img_A = np.expand_dims(img_A, axis=0)

                    for j, (img_B, label_B) in enumerate(data_loader_B):
                        if label_B[label] == 0: continue

                        img_B = np.expand_dims(img_B, axis=0)
                        fake_F = g_TF.predict(img_A)
                        fake_T = g_FT.predict(img_B)

                        dT_loss_real = d_T.train_on_batch(img_A, valid)
                        dT_loss_fake = d_T.train_on_batch(fake_T, fake)
                        dT_loss = 0.5 * np.add(dT_loss_real, dT_loss_fake)

                        dF_loss_real = d_F.train_on_batch(img_B, valid)
                        dF_loss_fake = d_T.train_on_batch(fake_F, fake)
                        dF_loss = 0.5 * np.add(dF_loss_real, dF_loss_fake)

                        d_loss = 0.5 * np.add(dT_loss, dF_loss)

                        # Train Gens
                        g_loss = self.GANs[label].train_on_batch([img_A, img_B], [valid, valid, img_A, img_B, img_A, img_B])

                        if d_loss[1] > 0.99:
                            d_T.trainable = False

                        print("\t\t[D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f]" % 
                                    (d_loss[0], 100*d_loss[1],
                                     g_loss[0],
                                     np.mean(g_loss[1:3]),
                                     np.mean(g_loss[3:5]),
                                     np.mean(g_loss[5:6])))
                    data_loader_B.__reset__()
                data_loader_A.__reset__()
            print('%s Storing Trained Models' % curr_time)
            self.save_models()
            print("\n\n")

        curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def test(self):
        self.load_weights()
        
        label_counts = dict()
        for i, x in enumerate(self.atts):
            os.makedirs('images/%s' % x, exist_ok=True)
            label_counts[i] = 0

        print('Loading dataset')
        data_loader_A = CelebDataLoader(self.data_dir, self.atts, method='test')
        data_loader_B = CelebDataLoader(self.data_dir, self.atts, method='test')
        print('Loaded dataset')
        
        for i, (img_A, label_A) in enumerate(data_loader_A):
            img_A = np.expand_dims(img_A, axis=0)
            for z, label in enumerate(label_A):
                if label == 1: continue
                for j, (img_B, label_B) in enumerate(data_loader_B):
                    if label_B[z] == 0: continue
                    print('Predicting')
                    img_B = np.expand_dims(img_B, axis=0)
                    label_counts[label] += 1
                    g_AB = self.g_TF_lst[z]
                    g_BA = self.g_FT_lst[z]

                    fake_B = g_AB.predict(img_A)
                    fake_A = g_BA.predict(img_B)

                    regen_A = g_BA.predict(fake_B)
                    regen_B = g_BA.predict(fake_A)

                    gen = np.concatenate([img_A, fake_B, regen_A, img_B, fake_A, regen_B])
                    gen = 0.5*gen + 0.5

                    titles = ['Original', 'Translated', 'Reconstructed']
                    fig, ax = plt.subplots(2,3)
                    cnt = 0
                    for w in range(2):
                        for y in range(3):
                            ax[w,y].imshow(gen[cnt])
                            ax[w,y].set_title(titles[y])
                            ax[w,y].axis('off')
                            cnt += 1
                    curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print('[%s] Storing image to: %s/%d_%d.png' % (curr_time, self.atts[z], label, label_counts[label]))
                    fig.savefig('images/%s/%d_%d.png' % (self.atts[z], label, label_counts[label]))
            data_loader_B.__reset__()
        print('Finished testing, images written in images/ folder')



