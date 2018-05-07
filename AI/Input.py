import pyautogui
import tensorflow as tf
import PIL
from PIL import Image
import numpy as np

class Sight():
    def __init__(self, rate=0.5):
        screen_size = pyautogui.screenshot().size
        self.im_size = [128, 128, 3]
        self.rate = rate
        self.concat_size = [screen_size[0], int(screen_size[1] / 2), 3]

    def make_image(self, name="object.jpg"):
        im = pyautogui.screenshot()
        im.save(name)

    def see(self):
        im = pyautogui.screenshot()

        return im

    def compress(self, im):
        width = int(self.im_size[0])
        height = int(self.im_size[1])
        im = im.resize((width, height), resample=PIL.Image.BILINEAR)
        im = im / 255.

        return im

    def batch(self, im):
        im = np.expand_dims(im, 0)  #batch를 요구해서 차원을 늘려서 처리함.
        im = im.transpose((0, 2, 1, 3))

        return im


    def get_image(self):    #스크린샷을 찍어 im_size크기로 배치를 만들어 반환
        width = int(self.im_size[0])
        height = int(self.im_size[1])

        im = pyautogui.screenshot()

        #size 변경
        im = im.resize((width, height), resample=PIL.Image.BILINEAR)

        #batch 추가
        im = np.expand_dims(im, 0)  #batch를 요구해서 차원을 늘려서 처리함.
        im = im.transpose((0, 2, 1, 3))


        im = im / 255.  #0~255 => 0~1

        return im

    def get_object(self, name="object.jpg"):
        width = int(self.im_size[0])
        height = int(self.im_size[1])

        im = Image.open(name)
        im = im.resize((width, height), resample=PIL.Image.BILINEAR)
        im = np.expand_dims(im, 0)
        im = im.transpose((0, 2, 1, 3))

        return im

    def concat(self, im1, im2):
        w1, h1 = im1.size
        w2, h2 = im2.size
        im = Image.new( "RGB", ( w1 + w2, max(h1, h2) ) )
        im.paste(im1)
        im.paste(im2, (w1, 0))
        return im

    def rgb2gray(self, im):
        return im.convert('L')

    def get_input(self):
        im_sc = self.get_image()
        print(im_sc.size)
        im_obj = self.get_object()
        print(im_obj.size)
        im_in = self.concat(im_sc, im_obj)
        print(im_in.size)
        im_in = np.expand_dims(im_in, 0)
        print(im_in.size)
        return im_in




if __name__ == '__main__':
    im = Sight()
    im1 = im.get_image()
    # im1 = im1 / 255.
    print(im1[0][0][60])
    # im.make_image()
    # im2 = im.get_object()
    # im3 = im.concat(im1, im2)
    # a = tf.image.decode_image(im3)
    # im3 = Image.new("RGB", (248, 248))
    # im3.paste(im1, (0,0))
    # im3.paste(im2, (124, 0))
    # im.get_input()

    # X = tf.placeholder(dtype=tf.float32)
    #
    # # You need to set the area to crop in boxes and resize it to in crop_size
    # Y = tf.image.crop_and_resize(X, boxes=[[.25, .25, .75, .75]], crop_size=[100, 100], box_ind=[0])
    #
    # sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()
    # out = sess.run(Y, {X: np.expand_dims(im3, 0)})
    # print(out)