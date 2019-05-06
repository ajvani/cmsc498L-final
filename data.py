import os
from PIL import Image

import numpy as np
import tensorflow as tf

class CelebData(object):
    def __init__(self, data_dir, target_atts, method='train'):
        # one hot encoding labels
        self.att_encoding = {
                '5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
                'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
                'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
                'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
                'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
                'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
                'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
                'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
                'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
                'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
                'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
                'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
                'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39 }

        # load labels, image names from csv
        attrs_file = os.path.join(data_dir, 'list_attr_celeba.csv')
        img_dir = os.path.join(data_dir, 'img_align_celeba')
        img_names = np.loadtxt(attrs_file, delimiter=',', skiprows=1, usecols=[0], dtype=np.str)
        img_paths = [os.path.join(img_dir, name) for name in img_names]
        att_cols = [self.att_encoding[att]+1 for att in target_atts]
        labels = np.loadtxt(attrs_file, delimiter=',', skiprows=1, usecols=att_cols, dtype=np.int64)

        def normalize(img_file,label):
            # resize img to (64,64) + normalize pixels in images to (-1,1)
            # normalize labels to (0,1)
            img = tf.read_file(img_file)
            img = tf.image.decode_jpeg(img, 3)
            img = tf.image.resize_images(img, [64,64], tf.image.ResizeMethod.BICUBIC)
            img = img / 127.5 - 1
            label = (label + 1) // 2
            return img, label

        # train/test/val split
        if method == 'test':
            drop_remainder = False
            repeat = 1
            img_paths = img_paths[182637:]
            labels = labels[182637:]
        elif method == 'val':
            img_paths = img_paths[182000:182637]
            labels = labels[182000:182637]
        else:
            img_paths = img_paths[:182000]
            labels = labels[:182000]

        # setup dataset
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        dataset = dataset.map(normalize)
        if method != 'test': dataset.shuffle(2048)

        # instance variables
        self._dataset = dataset
        self._num_img = len(img_paths)

    def __iterator__(self):
        self._iterator = self._dataset.make_initializable_iterator()
        return self._iterator

    def __num_imgs__(self):
        return self._num_img

    def __dataset__(self):
        return self._dataset

class DataTester():
    # test method to ensure data loaded properly
    def test(self):
        def denormalize(img):
            img = (img + 1) * 127.5
            img = np.uint8(img)
            return img
        
        test_atts = ['Bangs']
        data = CelebData('./data', test_atts, method='val')
        it = data.__iterator__()
        sample_batch = it.get_next()
        img = sample_batch[0]
        print(img.shape)
        
        sess = tf.Session()
        sess.run(it.initializer)
        with sess.as_default():
            eval_img = img.eval()
            _im_obj = denormalize(np.asarray(eval_img))
            print(_im_obj)
            im_obj = Image.fromarray(_im_obj)
            im_obj.show()
