# -*- coding: utf-8 -*-

'''
Layer for Linear regression output with hard negative mining 

PNratio: float.
The ratio of negative samples over positive samples.
When PNratio = 0, this is exactly the LinearRegressionOutput operator.
'''
import mxnet as mx


class neg_mining_regression(mx.operator.CustomOp):
    def __init__(self, neg_mult=9, hard_neg_ratio=0.5):
        super(neg_mining_regression, self).__init__()
        self.neg_mult = float(neg_mult)  # 负样本基于正样本的倍数
        self.hard_neg_ratio = float(hard_neg_ratio)  # 负样本中,hard负样本所占的比例
        print neg_mult, hard_neg_ratio

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pred = in_data[0]
        label = in_data[1]
        loss_raw = (pred - label)

        pos_flag = (label > 0.1) # 出来的flag数据类型是float32,不是bool型,不能直接当成标记使用[flag]
        # print 'max pos_flag: ', mx.ndarray.max(pos_flag.reshape((pos_flag.size, 1)), axis=0).asscalar()
        # print 'min pos_flag: ', mx.ndarray.min(pos_flag.reshape((pos_flag.size, 1)), axis=0).asscalar()
        # print 'pos_flag:', pos_flag.dtype, pos_flag.shape
        neg_flag = (label <= 0.1)
        # print 'neg_flag:', neg_flag.dtype, neg_flag.shape

        pos_num = mx.ndarray.sum(pos_flag).asscalar()
        neg_num = mx.ndarray.sum(neg_flag).asscalar()
        select_neg_num = min([pos_num * self.neg_mult, neg_num])
        select_hard_neg_num = int(self.hard_neg_ratio * select_neg_num)
        # print self.hard_neg_ratio
        # print select_hard_neg_num
        select_norm_neg_num = int(select_neg_num - select_hard_neg_num)
        # print select_norm_neg_num
        # 确定hard负样本的位置
        if select_hard_neg_num > 0:
            temp_loss = mx.ndarray.abs(loss_raw) # important !!!!!!!!!
            temp_loss *= neg_flag
            sorted_loss = mx.ndarray.sort(temp_loss.reshape((temp_loss.size, 1)), axis=0, is_ascend=False)
            # print sorted_loss.shape, sorted_loss[0].asscalar(), sorted_loss[1].asscalar(), sorted_loss[2].asscalar()
            threshold_value = sorted_loss[select_hard_neg_num].asscalar()
            # print 'threshold value: ', threshold_value
            # print 'max temp_loss: ', mx.ndarray.max(temp_loss.reshape((temp_loss.size, 1)), axis=0).asscalar()
            # print 'min temp_loss: ', mx.ndarray.min(temp_loss.reshape((temp_loss.size, 1)), axis=0).asscalar()
            select_hard_neg_flag = (temp_loss >= threshold_value)
        else:
            select_hard_neg_flag = mx.ndarray.zeros(label.shape, label.context, 'float32')
        # print 'max select_hard_neg_flag: ', mx.ndarray.max(select_hard_neg_flag.reshape((select_hard_neg_flag.size, 1)), axis=0).asscalar()
        # print 'min select_hard_neg_flag: ', mx.ndarray.min(select_hard_neg_flag.reshape((select_hard_neg_flag.size, 1)), axis=0).asscalar()
        # print 'select_hard_neg_flag:' , select_hard_neg_flag.dtype, select_hard_neg_flag.shape


        # 确定普通负样本的位置
        select_norm_neg_prob = float(select_norm_neg_num) / (neg_num - mx.ndarray.sum(select_hard_neg_flag).asscalar())
        random_select_norm_neg_mat = mx.ndarray.random_uniform(low=0, high=1, shape=label.shape, ctx=label.context)
        select_norm_neg_flag = (random_select_norm_neg_mat < select_norm_neg_prob)
        select_norm_neg_flag *= (1-pos_flag-select_hard_neg_flag)
        # print 'max select_norm_neg_flag: ', mx.ndarray.max(select_norm_neg_flag.reshape((select_norm_neg_flag.size, 1)), axis=0).asscalar()
        # print 'min select_norm_neg_flag: ', mx.ndarray.min(select_norm_neg_flag.reshape((select_norm_neg_flag.size, 1)), axis=0).asscalar()
        # print 'pos_flag: ', mx.ndarray.sum(pos_flag).asnumpy()[0]
        # print 'select_hard_neg_flag: ', mx.ndarray.sum(select_hard_neg_flag).asnumpy()[0]
        # print 'select_norm_neg_flag: ', mx.ndarray.sum(select_norm_neg_flag).asnumpy()[0]

        # temp_flag_1 = pos_flag + select_hard_neg_flag
        # print 'max temp_flag_1: ', mx.ndarray.max(temp_flag_1.reshape((temp_flag_1.size, 1)), axis=0).asscalar()
        # print 'min temp_flag_1: ', mx.ndarray.min(temp_flag_1.reshape((temp_flag_1.size, 1)), axis=0).asscalar()
        # temp_flag_2 = pos_flag + select_norm_neg_flag
        # print 'max temp_flag_2: ', mx.ndarray.max(temp_flag_2.reshape((temp_flag_2.size, 1)), axis=0).asscalar()
        # print 'min temp_flag_2: ', mx.ndarray.min(temp_flag_2.reshape((temp_flag_2.size, 1)), axis=0).asscalar()
        # temp_flag_3 = select_hard_neg_flag + select_norm_neg_flag
        # print 'max temp_flag_3: ', mx.ndarray.max(temp_flag_3.reshape((temp_flag_3.size, 1)), axis=0).asscalar()
        # print 'min temp_flag_3: ', mx.ndarray.min(temp_flag_3.reshape((temp_flag_3.size, 1)), axis=0).asscalar()
        active_flag = pos_flag + select_hard_neg_flag + select_norm_neg_flag
        # print 'active_flag: ', mx.ndarray.sum(active_flag).asnumpy()[0]
        # print 'max active_flag: ', mx.ndarray.max(active_flag.reshape((active_flag.size, 1)), axis=0).asscalar()
        # print 'min active_flag: ', mx.ndarray.min(active_flag.reshape((active_flag.size, 1)), axis=0).asscalar()
        loss = loss_raw / mx.ndarray.sum(active_flag) # 及其重要!!!!!!!!!!!!!!!之前被忽略了!!!!!!!-------但是这里其实还是一个定值!!!!!!!!!!!!!!
        loss *= active_flag
        self.assign(in_grad[0], req[0], loss)



@mx.operator.register("neg_mining_regression")
class neg_mining_regressionProp(mx.operator.CustomOpProp):
    def __init__(self, neg_mult=9, hard_neg_ratio=0.5):
        # print type(neg_mult), neg_mult
        # print type(hard_neg_ratio), hard_neg_ratio
        super(neg_mining_regressionProp, self).__init__(need_top_grad=False)
        self.neg_mult = neg_mult
        self.hard_neg_ratio = hard_neg_ratio

    def list_arguments(self):
        return ['pred', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):

        return neg_mining_regression(self.neg_mult, self.hard_neg_ratio)
