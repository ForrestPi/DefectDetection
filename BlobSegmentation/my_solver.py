# -*- coding: utf-8 -*-
import mxnet
import logging
import time
import os

class MySolver(object):
    '''
    '''
    def __init__(self, \
                 train_dataiter, \
                 symbol_net, \
                 net_initializer, \
                 net_optimizer_name, \
                 net_optimizer_params, \
                 data_names, \
                 label_names, \
                 context, \
                 nIter, \
                 net_train_metric, \
                 display_interval=10, \
                 eval_interval=100, \
                 eval_dataIter=None, \
                 net_eval_metric=None, \
                 num_eval_loop=1, \
                 model_file=None, \
                 save_prefix=None, \
                 save_start_index=0, \
                 model_save_interval=None, \
                 work_load=None):
        self.train_dataiter = train_dataiter
        self.eval_dataIter = eval_dataIter
        self.symbol_net = symbol_net
        self.net_initializer = net_initializer
        self.data_names = data_names
        self.label_names = label_names
        self.input_names = data_names+label_names
        self.context = context
        self.net_optimizer_name = net_optimizer_name
        self.net_optimizer_params = net_optimizer_params
        self.nIter = nIter
        self.net_train_metric = net_train_metric
        self.net_eval_metric = net_eval_metric
        self.display_interval = display_interval
        self.eval_interval = eval_interval
        self.save_prefix = save_prefix
        self.save_start_index = save_start_index
        assert self.save_start_index < self.nIter
        self.num_eval_loop = num_eval_loop
        self.model_file = model_file
        self.model_save_interval = model_save_interval
        self.work_load = work_load

        # init mxnet Module
        self.module = mxnet.module.Module(symbol=self.symbol_net, data_names=self.data_names, label_names=self.label_names, \
                                           context=self.context, work_load_list=self.work_load)

    def __init_module(self):
        arg_names = self.symbol_net.list_arguments() #这里的名字顺序决定了dataiter里面的数据存放顺序
        arg_shapes, out_shapes, aus_shapes = self.symbol_net.infer_shape()
        data_name_shape = [x for x in zip(arg_names, arg_shapes) if x[0] in self.data_names]
        label_name_shape = [x for x in zip(arg_names, arg_shapes) if x[0] in self.label_names]
        self.module.bind(data_shapes=data_name_shape, label_shapes=label_name_shape, for_training=True, grad_req='write')
        if self.model_file and self.model_file != '':
            self.load_checkpoint()
        else:
            self.module.init_params(initializer=self.net_initializer, allow_missing=True)

        self.module.init_optimizer(kvstore='device', optimizer=self.net_optimizer_name, optimizer_params=self.net_optimizer_params)



    # in this version, we do not use epoch concept. Each batch is randomly selected from the whole dataset
    def fit(self):
        logging.info('Initialize net -----------------------------------------\n')
        self.__init_module()

        logging.info('Start training in %s.--------------------------------------------\n',  str(self.context))

        sumTime = 0
        for i in xrange(self.save_start_index+1, self.nIter+1):
            start = time.time()
            
            # net forward and backward---------------------------------------------------------------------------------
            data_batch = self.train_dataiter.next(i)
            data_batch.as_ndarray(ctx=self.context[0])
            self.module.forward(data_batch=data_batch, is_train=True)

            self.module.backward()
            
            # update parameters----------------------------------------------------------------------------------------
            self.module.update()
            
            # update metric----------------------------------------------------------------------------------
            self.net_train_metric.update([data_batch.label[1]], self.module.get_outputs())

            
            sumTime += (time.time()-start)
            
            if self.display_interval>0 and i % self.display_interval == 0:

                name, value = self.net_train_metric.get()

                if i == 0:
                    logging.info('Init[0] %s loss: %04f. Time elapsed: %03fs.',  name, value, sumTime )
                else:
                    logging.info('Iter[%d] %s loss: %04f. Time elapsed: %03f s. Speed: %01f images/s.', \
                                 i, name, value, sumTime, self.display_interval*self.train_dataiter.getBatchsize()/sumTime)

                self.net_train_metric.reset()
                sumTime = 0    

            
            # evaluation-----------------------------------------------------------------------------------------------
            # if self.evalDataIter != None and i % self.evalInterval == 0:
            #
            #     for loop in xrange(self.numEvalLoop):
            #         evalBatch = self.evalDataIter.next()
            #         tempInputArr = {}
            #         tempInputArr[self.data_names[0]] = mxnet.nd.array(evalBatch.data, self.context)
            #         tempInputArr[self.label_names[0]] = mxnet.nd.array(evalBatch.label, self.context)
            #         self.eva_executor.copy_params_from(tempInputArr)
            #
            #         self.eva_executor.forward(is_train=False)
            #
            #         self.net_eval_metric.update([ tempInputArr[self.label_names[0]] ], \
            #            [self.eva_executor.outputs[0]])
            #     name, value = self.net_eval_metric.get()
            #     if value < best_validation_value:
            #         best_validation_value = value
            #         best_validation_iter = i
            #     logging.info('Iter[%d] <--Evaluation--> %s loss: %04f.->->->->->->->->->->->(Current best -- loss: %04f ; iter: %d)', i, name, value, best_validation_value, best_validation_iter)
            #     self.net_eval_metric.reset()
            
            # save checkpoint--------------------------------------------------------------------------------
            if i != 0 and i % self.model_save_interval == 0:
                self.save_checkpoint(i)

    def save_checkpoint(self, iterIdx):
        if self.save_prefix==None or self.save_prefix=='':
            logging.info('!!!!!Save prefix is not speficied. No save operations are taken.')
            return
        save_model_name = '%s_iter_%d_model.params' % (self.save_prefix, iterIdx+self.save_start_index)
        
        logging.info('<---------- Save checkpoint---------->')
        
        # save model params
        temp_arg_name_arrays, temp_aux_name_arrays = self.module.get_params()
        save_dict = {('arg:%s' % k) : v.as_in_context(mxnet.cpu()) for k, v in temp_arg_name_arrays.items() if k not in self.input_names}
        save_dict.update( {('aux:%s' % k) : v.as_in_context(mxnet.cpu()) for k, v in temp_aux_name_arrays.items()} )
        mxnet.nd.save(save_model_name, save_dict)
        logging.info('Iter[%d] <--Save params to file: %s-->', iterIdx, save_model_name)
           
    def load_checkpoint(self):
        logging.info('------>Load model from file: %s.\n', self.model_file)
        
        # load model params
        save_dict = mxnet.nd.load(self.model_file)
        
        arg_names = self.symbol_net.list_arguments()
        # get the arg shapes
        arg_shapes, out_shapes, aus_shapes = self.symbol_net.infer_shape()
        self.input_names = self.data_names+self.label_names
        
        arg_name_shapes = [x for x in zip(arg_names, arg_shapes)]
        self.arg_name_arrays = {k:mxnet.nd.zeros(s, self.context) for k, s in arg_name_shapes}
        self.arg_name_grads = {k:mxnet.nd.zeros(s, self.context) for k, s in arg_name_shapes if k not in self.input_names}
        self.arg_name_moms = {k:mxnet.nd.zeros(s, self.context) for k, s in arg_name_shapes if k not in self.input_names}
        self.aux_name_arrays = {} # currently nothing
        
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                self.arg_name_arrays.update({name:v.as_in_context(self.context)})
            if tp == 'aux':
                self.aux_name_arrays.update({name:v.as_in_context(self.context)})
        self.module.init_params(initializer=self.net_initializer,arg_params=self.arg_name_arrays,  aux_params=self.aux_name_arrays, allow_missing=True)
        
        
        
