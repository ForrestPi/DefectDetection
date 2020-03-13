# coding: utf-8
mxnet_095 = '/home/forrest/MXNet/mxnet-0.9.5/python'
import sys
sys.path.append(mxnet_095)
import mxnet

from my_symbol import construct_symbol_net
from my_prefetchingIter import ImageSegPrefetchingIter
from my_solver import MySolver
from my_metric import MSE
import my_logging
import logging
import time


def main():
    batch_size = 32
    # step 0 ---- init logging
    log_file = './log/train_' + time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())) + '.log'
    overwrite_flag = True
    my_logging.init_logging(log_file=log_file, overwrite_flag=overwrite_flag)
    logging.info('Mxnet Version: %s', str(mxnet.__version__))

    # step 1 ---- get symbol net, initializer
    net_symbol, net_initializer, net_data_names, net_label_names, net_lr_mult = \
        construct_symbol_net(batch_size)

    # step 2 ---- get data iteration
    nThread_1 = 5
    neg_bkg_ratio_list = [(10000, 2), (20000, 3), (30000,4), (40000, 5), (50000,6), (60000,7), (70000,8), (80000,9)]
    train_dataiter = ImageSegPrefetchingIter(dataPicklePath='./data/train_data_2017.6.22.pkl', \
                                                                      nThread=nThread_1, \
                                                                      batch_size=batch_size, \
                                                                      enable_horizon_flip=True, \
                                                                      neg_ratio_list=neg_bkg_ratio_list)

    # step 3 ---- get optimizer
    learning_rate = 1e-1
    weight_decay = 1e-4
    momentum = 0.9
    scheduler_step = 50000
    scheduler_factor = 0.5
    scheduler_lowest_lr = 1e-3

    lr_scheduler = mxnet.lr_scheduler.FactorScheduler(step=scheduler_step, factor=scheduler_factor, stop_factor_lr=scheduler_lowest_lr)
    optimizer_name = 'sgd'
    optimizer_params = {'learning_rate': learning_rate, \
                                            'wd': weight_decay, \
                                            'lr_scheduler': lr_scheduler, \
                                            'momentum': momentum}
                                            # 'lr_mult': net_lr_mult}

    # step 4 ---- get metric
    train_metric = MSE()
    eval_metric = None

    # step 5 ---- init solver and train
    nIter = 200000
    display_interval = 10
    eval_interval = 0
    ctx = [mxnet.gpu(0)]
    work_load = None
    model_file = ''  #
    save_prefix = './model/circle_location_seg_100x100'
    save_start_index = 0
    model_save_interval = 50
    my_solver = MySolver(train_dataiter, \
                         net_symbol, \
                         net_initializer, \
                         optimizer_name, \
                         optimizer_params, \
                         net_data_names, \
                         net_label_names, \
                         ctx,\
                         nIter, \
                         train_metric, \
                         display_interval=display_interval, \
                         eval_interval=eval_interval, \
                         eval_dataIter=None, \
                         net_eval_metric=eval_metric, \
                         num_eval_loop=1, \
                         model_file=model_file, \
                         save_prefix=save_prefix, \
                         save_start_index=save_start_index, \
                         model_save_interval=model_save_interval, \
                         work_load=work_load)
    my_solver.fit()
    quit()


if __name__ == '__main__':
    pass
    main()
