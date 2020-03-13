# coding: utf-8
mxnet_095 = '/home/forrest/MXNet/mxnet-0.9.5/python'
import sys
sys.path.append(mxnet_095)
import mxnet

import mxnet
import os
import numpy
import cv2
import cPickle
import random


class MyPredict(object):
    def __init__(self, symbolFile, modelFile):
        self.symbolFile = symbolFile
        self.modelFile = modelFile

    def loadModel(self, ctx, input_height, input_width):
        # load symbol first
        print '----> Load symbol and model from files: %s and %s.' % \
              (self.symbolFile, self.modelFile)

        if not os.path.exists(self.symbolFile):
            print 'The symbol file does not exist!!!!'
            quit()
        if not os.path.exists(self.modelFile):
            print 'The model file does not exist!!!!'
            quit()

        self.ctx = ctx
        self.input_height = input_height
        self.input_width = input_width

        self.symbolNet = mxnet.symbol.load(self.symbolFile)
        #        arg_names = self.symbolNet.list_arguments()
        #        arg_shapes, out_shapes, aus_shapes = self.symbolNet.infer_shape()
        #        arg_name_shapes = [x for x in zip(arg_names, arg_shapes)]


        save_dict = mxnet.nd.load(self.modelFile)
        self.arg_name_arrays = {}
        self.arg_name_arrays['data'] = mxnet.nd.zeros((1, 3, input_height, input_width), self.ctx)
        self.arg_name_arrays['label'] = mxnet.nd.zeros((1, 1, input_height, input_width), self.ctx)
        self.aux_name_arrays = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                self.arg_name_arrays.update({name: v.as_in_context(self.ctx)})
            if tp == 'aux':
                self.aux_name_arrays.update({name: v.as_in_context(self.ctx)})

        self.executor = self.symbolNet.bind(ctx=self.ctx, \
                                            args=self.arg_name_arrays, \
                                            grad_req='null', \
                                            aux_states=self.aux_name_arrays)

        print '----> Model is loaded successfully.'

    def predict(self, image, threshold, rawFlag=False):
        # minus means
        image_resized = cv2.resize(image, (self.input_width, self.input_height))

        image_resized = image_resized.astype(dtype=numpy.float32)
        image_resized[:, :, 0] = image_resized[:, :, 0] - 104.008
        image_resized[:, :, 1] = image_resized[:, :, 1] - 116.669
        image_resized[:, :, 2] = image_resized[:, :, 2] - 122.675

        image_resized = image_resized[:, :, :, numpy.newaxis]
        image_resized = image_resized.transpose([3, 2, 0, 1])

        self.executor.copy_params_from({'data': mxnet.nd.array(image_resized, self.ctx)})

        self.executor.forward(is_train=False)

        outputs = self.executor.outputs[0].asnumpy()
        outputs = numpy.squeeze(outputs)
        # 有三种插值方法效果会好一些  INTER_LINEAR（最快）    INTER_CUBIC   INTER_LANCZOS4
        outputs = cv2.resize(outputs, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        if rawFlag:
            return outputs

        out = numpy.zeros(outputs.shape, dtype=numpy.uint8)
        out[outputs > threshold] = 255

        return out


    def predict_lcr(self, image, threshold, rawFlag=False):
        # minus means
        image_resized = cv2.resize(image, (self.input_width, self.input_height))

        image_resized = image_resized.astype(dtype=numpy.float32)
        image_resized[:, :, 0] = image_resized[:, :, 0] - 104.008
        image_resized[:, :, 1] = image_resized[:, :, 1] - 116.669
        image_resized[:, :, 2] = image_resized[:, :, 2] - 122.675

        image_resized = image_resized[:, :, :, numpy.newaxis]
        image_resized = image_resized.transpose([3, 2, 0, 1])

        self.executor.copy_params_from({'data': mxnet.nd.array(image_resized, self.ctx)})

        self.executor.forward(is_train=False)

        outputs = self.executor.outputs[0].asnumpy()
        outputs = numpy.squeeze(outputs)
        if rawFlag:
            outputs = outputs*255
            return outputs.astype(dtype=numpy.uint8)

        temp = numpy.zeros(outputs.shape, dtype=numpy.uint8)
        temp[outputs>threshold] = 255
        temp = get_largest_connected_region(temp)

        return temp


def get_largest_connected_region(inputImage):
    img = inputImage.copy()
    contours, hierarchy = cv2.findContours(inputImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)

    del contours[numpy.argmax(areas)]
    temp = numpy.zeros((img.shape[0], img.shape[1]), dtype=numpy.uint8)
    cv2.fillPoly(temp, contours, color=(255), lineType=4)
    img[temp == 255] = 0

    return img

def test_trainset():
    compressedData = cPickle.load(open('./data/train_data_2017.6.22.pkl', 'rb'))
    predictor = MyPredict('./model/circle_location_seg_100x100_deploy.json', \
                          './model/circle_location_seg_100x100_iter_1000000_model.params')
    input_height = 100
    input_width = 100
    predictor.loadModel(mxnet.gpu(), 100, 100)

    for sample in compressedData:
        img = cv2.imdecode(sample['image'], cv2.CV_LOAD_IMAGE_COLOR)
        gt = cv2.imdecode(sample['gt'], cv2.CV_LOAD_IMAGE_GRAYSCALE)
        border = sample['border']
        img_height, img_width = img.shape[:2]
        gt_center_y = round((border[0]+border[1])/2.0)
        gt_center_x = round((border[2]+border[3])/2.0)

        change_ratio = 0.3
        height_change = int( (round(input_height/2.0)-round((border[1]-border[0])/2.0))*change_ratio )
        width_change = int( (round(input_width/2.0)-round((border[3]-border[2])/2.0))*change_ratio )
        crop_top_change = random.randint(-height_change, height_change)
        crop_left_change = random.randint(-width_change, width_change)

        crop_top = gt_center_y-round(input_height/2.0)+crop_top_change
        crop_left = gt_center_x-round(input_width/2.0)+crop_left_change
        if crop_top < 0:
            to_top = -crop_top
            from_top = 0
        else:
            to_top = 0
            from_top = crop_top
        if crop_top+input_height > img_height:
            to_bottom = img_height-crop_top
            from_bottom = img_height
        else:
            to_bottom = input_height
            from_bottom = crop_top+input_height
        if crop_left < 0:
            to_left = -crop_left
            from_left = 0
        else:
            to_left = 0
            from_left = crop_left
        if crop_left+input_width > img_width:
            to_right = img_width-crop_left
            from_right = img_width
        else:
            to_right = input_width
            from_right = crop_left+input_width

        img_crop = numpy.zeros((input_height, input_width, 3), dtype=numpy.uint8)
        gt_crop = numpy.zeros((input_height, input_width), dtype=numpy.uint8)
        img_crop[int(to_top):int(to_bottom), int(to_left):int(to_right), :] = img[int(from_top):int(from_bottom), int(from_left):int(from_right), :]
        gt_crop[int(to_top):int(to_bottom), int(to_left):int(to_right)] = gt[int(from_top):int(from_bottom), int(from_left):int(from_right)]

        predicted = predictor.predict_lcr(img_crop, 0.6,rawFlag=True)

        cv2.imshow('im', img_crop)
        cv2.imshow('gt', gt_crop)
        cv2.imshow('predicted', predicted)
        cv2.waitKey()
    cv2.destroyAllWindows()

def test_testset():
    root = './test_image/'
    image_path_list = []
    for parent, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if not filename.lower().endswith('jpg'):
                continue
            image_path_list.append(os.path.join(parent, filename))

    predictor = MyPredict('./model/circle_location_seg_100x100_deploy.json', \
                          './model/circle_location_seg_100x100_iter_1000000_model.params')
    input_height = 100
    input_width = 100
    predictor.loadModel(mxnet.gpu(), 100, 100)
    for file_path in image_path_list:
        img = cv2.imread(file_path, cv2.CV_LOAD_IMAGE_COLOR)
        img_height, img_width = img.shape[:2]

        center_y = round(img_height/2.0)
        center_x = round(img_width/2.0)

        crop_top = center_y-round(input_height/2.0)
        crop_left = center_x-round(input_width/2.0)
        if crop_top < 0:
            to_top = -crop_top
            from_top = 0
        else:
            to_top = 0
            from_top = crop_top
        if crop_top+input_height > img_height:
            to_bottom = img_height-crop_top
            from_bottom = img_height
        else:
            to_bottom = input_height
            from_bottom = crop_top+input_height
        if crop_left < 0:
            to_left = -crop_left
            from_left = 0
        else:
            to_left = 0
            from_left = crop_left
        if crop_left+input_width > img_width:
            to_right = img_width-crop_left
            from_right = img_width
        else:
            to_right = input_width
            from_right = crop_left+input_width

        img_crop = numpy.zeros((input_height, input_width, 3), dtype=numpy.uint8)
        img_crop[int(to_top):int(to_bottom), int(to_left):int(to_right), :] = img[int(from_top):int(from_bottom), int(from_left):int(from_right), :]

        predicted = predictor.predict_lcr(img_crop, 0.6,rawFlag=True)

        cv2.imshow('im', img_crop)
        cv2.imshow('predicted', predicted)
        cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_trainset()
    test_testset()
