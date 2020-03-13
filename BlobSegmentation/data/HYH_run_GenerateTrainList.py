# -*- coding: utf-8 -*-
import numpy
import cv2
import os
import os.path
import shutil


def largestConnectedRegion(inputImage):
    img = inputImage.copy()
    contours, hierarchy = cv2.findContours(inputImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours == []:
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


def generate_train_list():
    imRoots = ['./hole/photo100/']#, './hole/20170613/'
    gtRoots = ['./hole/photo100biao/']#, './hole/20170613biao/'

    imagePathList = []
    imageNameList = []
    gtPathList = []
    gtNameList = []
    for imRoot in imRoots:
        for parent, dirnames, filenames in os.walk(imRoot):
            for filename in filenames:
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
                    imagePathList.append(os.path.join(parent, filename))
                    imageNameList.append(filename)
    for gtRoot in gtRoots:
        for parent, dirnames, filenames in os.walk(gtRoot):
            for filename in filenames:
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                    gtPathList.append(os.path.join(parent, filename))
                    gtNameList.append(filename)

    trainIndex = range(len(imagePathList))

    trainListFile = open('train_list_2017.6.22.txt', 'a')

    for i in xrange(len(trainIndex)):
        image = cv2.imread(imagePathList[trainIndex[i]])
        print imagePathList[trainIndex[i]]
        if imageNameList[trainIndex[i]] not in gtNameList:
            print 'Image %s has no gt. Exit' % (imagePathList[trainIndex[i]])
            quit()
        print gtPathList[gtNameList.index(imageNameList[trainIndex[i]])]
        mask = cv2.imread(gtPathList[gtNameList.index(imageNameList[trainIndex[i]])], cv2.IMREAD_GRAYSCALE) #
        if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
            print 'Image %s has different size compared to its gt. Exit' % (imagePathList[trainIndex[i]])
            quit()
        mask[mask > 127] = 255
        mask[mask <= 127] = 0

        mask = largestConnectedRegion(mask)
        foreground = numpy.where(mask == 255)
        left = foreground[1].min()
        right = foreground[1].max()
        top = foreground[0].min()
        bottom = foreground[0].max()
        if right - left <= 0 or bottom - top <= 0:
            print 'Image %s has too small gt. Exit' % (imagePathList[trainIndex[i]])
            continue

        lineStr = '%s %s %d %d %d %d\n' % (
            imagePathList[trainIndex[i]], gtPathList[gtNameList.index(imageNameList[trainIndex[i]])], \
            top, bottom, left, right)
        trainListFile.writelines(lineStr)

        print 'Image %s ' % (imagePathList[trainIndex[i]])
        print 'process: %d' % (i)

    trainListFile.flush()
    trainListFile.close()


def inverse_BW():
    gtRoots = ['./hole/20170613biao/', './hole/photo100biao/']
    gtPathList = []
    gtNameList = []
    for gtRoot in gtRoots:
        for parent, dirnames, filenames in os.walk(gtRoot):
            for filename in filenames:
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                    gtPathList.append(os.path.join(parent, filename))
                    gtNameList.append(filename)

    for gt_path in gtPathList:
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = gt.astype(dtype=numpy.float32)
        gt = numpy.abs(gt - 255)
        gt = gt.astype(dtype=numpy.uint8)

        cv2.imwrite(gt_path, gt)


if __name__ == '__main__':
# inverse_BW()
    generate_train_list()
