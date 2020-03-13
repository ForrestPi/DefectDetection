# -*- coding: utf-8 -*-
# 该版本的输入尺寸为50x50
mxnet_095 = '/home/forrest/MXNet/mxnet-0.9.5/python'
import sys
sys.path.append(mxnet_095)
import mxnet
import cPickle
import numpy
import random
import threading
import cv2
import time
import logging
import data_augmentation_util


class MyDataBatch():
    def __init__(self):
        self.data = []
        self.label = []

    def append_data(self, new_data):
        self.data.append(new_data)

    def append_label(self, new_label):
        self.label.append(new_label)

    def as_ndarray(self, ctx):
        for i in range(len(self.data)):
            self.data[i] = mxnet.ndarray.array(self.data[i], ctx=ctx)
        for i in range(len(self.label)):
            self.label[i] = mxnet.ndarray.array(self.label[i], ctx=ctx)

    @property
    def data(self):
        return self.data

    @property
    def label(self):
        return self.label

def test_MyDataBatch():
    temp_databatch = MyDataBatch()
    temp_databatch.append_data(123)
    temp_databatch.append_label(123)
    print temp_databatch.data, temp_databatch.label

class ImageSegPrefetchingIter(mxnet.io.DataIter):
    def __init__(self, dataPicklePath, \
                 nThread, \
                 batch_size, \
                 enable_horizon_flip=False, \
                 neg_ratio_list=None  # [(iter, ratio),()]
                 ):
        # read data from pickle files --------------
        try:
            logging.info('Start to load data pickle files, it may cost several minutes----------.\n')
            self.compressedData = cPickle.load(open(dataPicklePath, 'rb'))
            logging.info('Pickle files are loaded.\n')
        except:
            logging.error('Error ocurrs when loading pickl files!!!')
            quit()
        self.num_sample = len(self.compressedData)
        logging.info('There are %d samples.\n', self.num_sample)

        self.nThread = nThread
        self.batch_size = batch_size
        self.enable_horizon_flip = enable_horizon_flip
        self.B_mean = 104.008
        self.G_mean = 116.669
        self.R_mean = 122.675
        self.input_height = 100
        self.input_width = 100
        self.neg_ratio = 1
        neg_ratio_list.reverse()
        self.neg_ratio_list = neg_ratio_list

        # prepare threads
        self.data_taken = [threading.Event() for i in range(self.nThread)]
        for e in self.data_taken:
            e.set()
        self.data_ready = [threading.Event() for i in range(self.nThread)]
        for e in self.data_ready:
            e.clear()
        self.data_queue = [None for i in range(self.nThread)]
        self.current_batch = None
        self.started = True

        # the main procedure running in a thread
        def prefetch_func(self, i):
            while True:
                timeoutFlag = self.data_taken[i].wait()  # here, the arg timeout is to detect the exit or thread reduntant
                if not self.started or not timeoutFlag:
                    break

                try:
                    self.data_queue[i] = self._prepare_batch(i)
                except Exception, e:
                    logging.error('Prepare batch wrong in thread %d !! -------------> \n %s', i, e.message)
                    self.data_queue[i] = None
                    continue

                self.data_taken[i].clear()
                self.data_ready[i].set()

        self.prefetch_threads = [threading.Thread(target=prefetch_func, args=[self, i]) for i in range(self.nThread)]

        for thread in self.prefetch_threads:
            thread.setDaemon(True)  # make it clear ~~~~~~~~~~~~~~~~~~
            thread.start()

    def iter_next(self, num_iter):
        #set neg ratio
        for item in self.neg_ratio_list:
            if num_iter >= item[0]:
                self.neg_ratio = item[1]
                break
        # keep looping until getting the databatch
        while True:
            for i, dataBatch in enumerate(self.data_queue):
                if not self.started:
                    quit()
                if dataBatch is None:
                    continue

                self.data_ready[i].wait()
                self.current_batch = dataBatch
                self.data_queue[i] = None
                self.data_ready[i].clear()
                self.data_taken[i].set()

                return True
            time.sleep(0.001)  # key part!!!!!!!!!!!!

    def __del__(self):
        # relase all threads
        self.started = False
        for e in self.data_taken:
            e.set()

        for thread in self.prefetch_threads:
            thread.join(1)

    # this fun is to prepare data batch including data augmentation, and return a DataBatch
    def _prepare_batch(self, i):
        im_batch = numpy.zeros((self.batch_size, 3, self.input_height, self.input_width), dtype=numpy.float32)
        gt_batch = numpy.zeros((self.batch_size, 1, self.input_height, self.input_width), dtype=numpy.float32)
        data_batch = MyDataBatch()

        for loop in xrange(self.batch_size):

            rand_idx = random.randint(0, self.num_sample-1)

            # get image and gt
            img = cv2.imdecode(self.compressedData[rand_idx]['image'].copy(), cv2.CV_LOAD_IMAGE_COLOR)
            gt = cv2.imdecode(self.compressedData[rand_idx]['gt'].copy(), cv2.CV_LOAD_IMAGE_GRAYSCALE)
            border = self.compressedData[rand_idx]['border'].copy() # numpy.array([top, bottom, left, right])

            img_height, img_width = img.shape[:2]
            gt_center_y = round((border[0]+border[1])/2.0)
            gt_center_x = round((border[2]+border[3])/2.0)

            change_ratio = 0.3
            height_change = int( (round(self.input_height/2.0)-round((border[1]-border[0])/2.0))*change_ratio )
            width_change = int( (round(self.input_width/2.0)-round((border[3]-border[2])/2.0))*change_ratio )
            crop_top_change = random.randint(-height_change, height_change)
            crop_left_change = random.randint(-width_change, width_change)

            crop_top = gt_center_y-round(self.input_height/2.0)+crop_top_change
            crop_left = gt_center_x-round(self.input_width/2.0)+crop_left_change
            if crop_top < 0:
                to_top = -crop_top
                from_top = 0
            else:
                to_top = 0
                from_top = crop_top
            if crop_top+self.input_height > img_height:
                to_bottom = img_height-crop_top
                from_bottom = img_height
            else:
                to_bottom = self.input_height
                from_bottom = crop_top+self.input_height
            if crop_left < 0:
                to_left = -crop_left
                from_left = 0
            else:
                to_left = 0
                from_left = crop_left
            if crop_left+self.input_width > img_width:
                to_right = img_width-crop_left
                from_right = img_width
            else:
                to_right = self.input_width
                from_right = crop_left+self.input_width

            img_crop = numpy.zeros((self.input_height, self.input_width, 3), dtype=numpy.uint8)
            gt_crop = numpy.zeros((self.input_height, self.input_width), dtype=numpy.uint8)
            img_crop[int(to_top):int(to_bottom), int(to_left):int(to_right), :] = img[int(from_top):int(from_bottom), int(from_left):int(from_right), :]
            gt_crop[int(to_top):int(to_bottom), int(to_left):int(to_right)] = gt[int(from_top):int(from_bottom), int(from_left):int(from_right)]
            if self.enable_horizon_flip and random.random() < 0.5:
                img_crop = data_augmentation_util.horizon_flip(img_crop)
                gt_crop = data_augmentation_util.horizon_flip(gt_crop)
            # cv2.imshow('im', img_crop)
            # cv2.imshow('gt', gt_crop)
            # cv2.waitKey()
            img_crop = img_crop.astype(dtype=numpy.float32)
            gt_crop = gt_crop.astype(dtype=numpy.float32)

            img_crop[:, :, 0] = img_crop[:, :, 0] - self.B_mean
            img_crop[:, :, 1] = img_crop[:, :, 1] - self.G_mean
            img_crop[:, :, 2] = img_crop[:, :, 2] - self.R_mean
            gt_crop = gt_crop/255.0

            im_batch[loop, :, :, :] = img_crop.transpose([2, 0, 1])
            gt_batch[loop, 0, :, :] = gt_crop


        # doing sample balance
        pos_flag = gt_batch>0.5
        num_pos = numpy.sum(pos_flag)
        num_neg = gt_batch.size-num_pos

        select_num_neg = min([num_pos*self.neg_ratio, num_neg])
        prob_threshold = float(select_num_neg)/num_neg

        prob_mat = numpy.random.random(gt_batch.shape)
        prob_mat[pos_flag] = 1
        mask_batch = numpy.zeros(gt_batch.shape, dtype=numpy.bool)
        mask_batch[prob_mat<prob_threshold] = 1
        mask_batch[pos_flag] = 1

        data_batch.append_data(im_batch)
        data_batch.append_label(mask_batch)
        data_batch.append_label(gt_batch)
        return data_batch

    def reset(self):
        pass

    def getBatchsize(self):
        return self.batch_size

    def next(self, num_iter):
        pass
        if self.iter_next(num_iter):
            return self.current_batch

    def getdata(self):
        pass
        return self.current_batch.data

    def getlabel(self):
        pass
        return self.current_batch.label

    def getindex(self):
        pass
        return self.current_batch.index

    def getpad(self):
        pass
        return self.current_batch.pad


def testPeopleSegPrefetchingIter():
    logging.getLogger().setLevel(logging.DEBUG)
    neg_ratio_list=[(10,2), (100,10), (200,20)]
    myIter = ImageSegPrefetchingIter(dataPicklePath='./data/train_data_2017.6.22.pkl', \
                                     nThread=5, \
                                     batch_size=32, \
                                     enable_horizon_flip=True,\
                                     neg_ratio_list=neg_ratio_list)

    numFetch = 1000
    start = time.time()
    for loop in range(numFetch):
        data_batch = myIter.next(loop)
        print loop

    print 'tatol fetching time: %f s' % (time.time() - start)


if __name__ == '__main__':
    testPeopleSegPrefetchingIter()
    # test_MyDataBatch()