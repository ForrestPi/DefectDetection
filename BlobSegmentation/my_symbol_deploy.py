# -*- coding: utf-8 -*-
mxnet_095 = '/home/forrest/MXNet/mxnet-0.9.5/python'
import sys
sys.path.append(mxnet_095)
import mxnet

# return both symbol and initializers
def construct_symbol_net():

    
    # batch size can be set here
    input = mxnet.symbol.Variable(name='data')
    
    # conv block 1 100x100-->100x100---------------------------------------------------------------------------------------
    conv1 = mxnet.symbol.Convolution(data=input, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=32, name='conv1')
    conv1 = mxnet.symbol.Activation(data=conv1, act_type='relu', name='relu_conv1')
    
    # conv block 2 100x100-->50x50----------------------------------------------------------------------------------------
    conv2 = mxnet.symbol.Convolution(data=conv1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_filter=64, name='conv2')
    conv2 = mxnet.symbol.Activation(data=conv2, act_type='relu', name='relu_conv2')
    
    # conv block 3 50x50-->25x25----------------------------------------------------------------------------------------
    conv3 = mxnet.symbol.Convolution(data=conv2, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_filter=64, name='conv3')
    conv3 = mxnet.symbol.Activation(data=conv3, act_type='relu', name='relu_conv3')
    
    # conv block 4 25x25-->13x13----------------------------------------------------------------------------------------
    conv4 = mxnet.symbol.Convolution(data=conv3, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_filter=80, name='conv4')
    conv4 = mxnet.symbol.Activation(data=conv4, act_type='relu', name='relu_conv4')
    
    # conv block 5 13x13-->7x7----------------------------------------------------------------------------------------
    conv5 = mxnet.symbol.Convolution(data=conv4, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_filter=80, name='conv5')
    conv5 = mxnet.symbol.Activation(data=conv5, act_type='relu', name='relu_conv5')
    
    # conv block 6 7x7-->4x4----------------------------------------------------------------------------------------
    conv6 = mxnet.symbol.Convolution(data=conv5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_filter=128, name='conv6')
    conv6 = mxnet.symbol.Activation(data=conv6, act_type='relu', name='relu_conv6')
    
    # conv block 7 4x4-->1x1----------------------------------------------------------------------------------------
    conv7 = mxnet.symbol.Convolution(data=conv6, kernel=(4, 4), stride=(1, 1), pad=(0, 0), num_filter=128, name='conv7')
    conv7 = mxnet.symbol.Activation(data=conv7, act_type='relu', name='relu_conv7')
    
    # deconv block 1 1x1-->4x4----------------------------------------------------------------------------------------
    deconv1 = mxnet.symbol.Deconvolution(data=conv7, kernel=(4, 4), stride=(1, 1), pad=(0, 0), num_filter=128, name='deconv1', no_bias=False)
    deconv1 = mxnet.symbol.Activation(data=deconv1, act_type='relu', name='relu_deconv1')
    # fusion block 1 4x4-->stack two feature maps
    fusion1 = mxnet.symbol.Concat(deconv1, conv6, dim=1, name='fusion1')
    
    # deconv block 2 4x4-->7x7----------------------------------------------------------------------------------------
    deconv2 = mxnet.symbol.Deconvolution(data=fusion1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_filter=80, name='deconv2', no_bias=False)
    deconv2 = mxnet.symbol.Activation(data=deconv2, act_type='relu', name='relu_deconv2')
    # fusion block 2 7x7-->stack two feature maps
    fusion2 = mxnet.symbol.Concat(deconv2, conv5, dim=1, name='fusion2')
    
    # deconv block 3 7x7-->13x13----------------------------------------------------------------------------------------
    deconv3 = mxnet.symbol.Deconvolution(data=fusion2, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_filter=80, name='deconv3', no_bias=False)
    deconv3 = mxnet.symbol.Activation(data=deconv3, act_type='relu', name='relu_deconv3')
    # fusion block 3 13x13-->stack two feature maps
    fusion3 = mxnet.symbol.Concat(deconv3, conv4, dim=1, name='fusion3')
    
    # deconv block 4 13x13-->25x25----------------------------------------------------------------------------------------
    deconv4 = mxnet.symbol.Deconvolution(data=fusion3, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_filter=64, name='deconv4', no_bias=False)
    deconv4 = mxnet.symbol.Activation(data=deconv4, act_type='relu', name='relu_deconv4')
    # fusion block 4 25x25-->stack two feature maps
    fusion4 = mxnet.symbol.Concat(deconv4, conv3, dim=1, name='fusion4')
    
    # deconv block 5 25x25-->50x50----------------------------------------------------------------------------------------
    deconv5 = mxnet.symbol.Deconvolution(data=fusion4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=64, name='deconv5', no_bias=False)
    deconv5 = mxnet.symbol.Activation(data=deconv5, act_type='relu', name='relu_deconv5')
    # fusion block 5 50x50-->stack two feature maps
    fusion5 = mxnet.symbol.Concat(deconv5, conv2, dim=1, name='fusion5')
    
    # deconv block 6 50x50-->100x100----------------------------------------------------------------------------------------
    deconv6 = mxnet.symbol.Deconvolution(data=fusion5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=128, name='deconv6', no_bias=False)
    deconv6 = mxnet.symbol.Activation(data=deconv6, act_type='relu', name='relu_deconv6')
    
    # loss block----------------------------------------------------------------------------------------
    predict = mxnet.symbol.Convolution(data=deconv6, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=1, name='predict_conv')
    predict = mxnet.symbol.Activation(data=predict, act_type='sigmoid', name='sigmoid_predict_conv')

    return predict
    
def test():
    my_symbol= construct_symbol_net()
    my_symbol.save('./model/circle_location_seg_100x100_deploy_080.json')

if __name__ == '__main__':
    pass
    test()
    
    
    
