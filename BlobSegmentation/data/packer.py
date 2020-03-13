#_*_ coding: utf-8 -*-


import cPickle
import cv2
import os
import numpy


"""
This class is for packing image data used for training and validation by using cPickle(serialization) and cv2.imencode
"""
class PackData(object):
    '''
    listFile: str , list file containing all im and gt with bbox
    saveLocation: str , example './save/xxx.pkl'
    compressParams: ('compress_type',quality), compress_type now only supports .jpg (or .jpeg); quality is 0-100 to balance compress rate
    '''
    def __init__(self, listFile, saveLocation, compressParams=('.jpg', 100)):
        self.listFile = listFile
        self.compressParams = compressParams
        self.saveLocation = saveLocation
    
    def setCompressParams(self, compressParams):
        self.compressParams = compressParams
    
    def pack(self):
        dataCollection = []
        imageNoneList=[]
        gtNoneList=[]
        imageComFailList=[]
        gtComFailList=[]
        successCount=0
        
        fin = open(self.listFile, 'r')
        count = 0

        for line in fin:
            count += 1
            tempData = {}
            line = line.replace('\n','').split(' ')
            imPath = line[0]
            gtPath = line[1]
            top = int(line[2])
            bottom = int(line[3])
            left = int(line[4])
            right = int(line[5])
            
            # load img
            img = cv2.imread(imPath)
            if img == None:
                print 'Image: %s  does not exist!!' % (imPath)
                imageNoneList.append(imPath)
                continue
            
            ret, buf = cv2.imencode(self.compressParams[0], img, [cv2.IMWRITE_JPEG_QUALITY, self.compressParams[1]])
            if not ret:
                print 'Image: %s compression error!!' % (imPath)
                imageComFailList.append(imPath)
                continue
            tempData['image'] = buf
            
            # load gt
            if gtPath == 'null':

                tempData['gt'] = None

            else:

                gt = cv2.imread(gtPath)
                if gt == None:
                    print 'GT: %s  does not exist!!' % (gtPath)
                    gtNoneList.append(gtPath)
                    continue
                if gt.shape != img.shape:
                    print 'GT and img have different shape!!'+str(gt.shape)+'|'+str(img.shape)
                    continue
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
                ret, buf = cv2.imencode(self.compressParams[0], gt, [cv2.IMWRITE_JPEG_QUALITY, self.compressParams[1]])
                if not ret:
                    print 'GT: %s compression error!!' % (gtPath)
                    gtComFailList.append(gtPath)
                    continue
                tempData['gt'] = buf
                # append bbox
                tempData['border'] = numpy.array([top, bottom, left, right])
                
            successCount +=1
            dataCollection.append(tempData)
            
            print '( %d ) Sample processed successfully.' % (successCount)

        
        print 'Processing statistics:'
        print 'There are %d image failed to read----' % (len(imageNoneList))
        for imagePath in imageNoneList:
            print imagePath
        print 'There are %d gt failed to read----' % (len(gtNoneList))
        for gtPath in gtNoneList:
            print gtPath
        print 'There are %d image failed to compress----' % (len(imageComFailList))
        for imagePath in imageComFailList:
            print imagePath
        print 'There are %d gt failed to compress----' % (len(gtComFailList))
        for gtPath in imageComFailList:
            print gtPath
        
        print 'start to save pickle file......'
        locationDir = os.path.dirname(self.saveLocation)
        if not os.path.exists(locationDir):
            os.makedirs(locationDir)
        cPickle.dump(dataCollection, open(self.saveLocation, 'wb'), cPickle.HIGHEST_PROTOCOL)
        print 'pickle file save successfully!'
        


def runPackData():
    dataListFileName = './train_list_2017.6.22.txt'
    packData = PackData(dataListFileName, './train_data_2017.6.22.pkl')
    packData.pack()


if __name__=='__main__':
    runPackData()
