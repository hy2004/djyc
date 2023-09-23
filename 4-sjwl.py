from typing import Any, Union
import pandas as pd
import numpy as np
from sklearn import metrics
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pywt
from wrocef import wavedec, wrcoef
import sklearn.cluster as skc
import time
from sklearn import metrics  # 评估模型
import math


# 对称平均绝对百分比误差（SMAPE）
def smape(PD, RD):
    return 2.0 * np.mean(np.abs(PD - RD) / (np.abs(PD) + np.abs(RD))) * 100

class BPNetwork():
    def __init__(self, x_d, y_d, xx_d, y2, max_price, min_price, max_price1, min_price1):
        self.x_d = x_d
        self.y_d = y_d
        self.xx_d = xx_d
        self.y2 = y2

    # 定义层
    def fc_layer(self, input, in_node, out_node, activation_fun=None):
        # 参数初始化  权重参数使用初始化
        # weight = np.random.randn(in_node, out_node) / np.sqrt(in_node / 2)
        # weight = tf.Variable(weight, dtype=tf.float32)
        # bias = tf.Variable(tf.add(tf.zeros(shape=[num_sample, out_node], dtype=tf.float32), 0.1), dtype=tf.float32)
        shape = [in_node, out_node]
        bias = tf.Variable(tf.zeros(out_node))
        initial = tf.truncated_normal(shape, stddev=0.1)
        weight = tf.Variable(initial)
        y = tf.nn.tanh(tf.matmul(input, weight) + bias)

        return y



    def BPNN(self):
        # 定义占位变量
        x_node = tf.placeholder(tf.float32, shape=[None, 9])
        y_node = tf.placeholder(tf.float32, shape=[None, 1])
        num_sample = 6  # 分批集的大小
        # 定义神经网络模型
        layer1 = self.fc_layer(x_node, 9, 30, tf.nn.tanh)
        layer2 = self.fc_layer(layer1, 30, 8, tf.nn.tanh)
        y_pred = self.fc_layer(layer2, 8, 1, num_sample)

        # 定义损失函数和优化算法
        train_loss = tf.losses.mean_squared_error(labels=y_node, predictions=y_pred)
        # 损失函数：在训练过程中增加了平方和损失.( labels：真实的输出张量 , predictions：预测的输出.)
        # 在这个函数中，weights作为loss的系数.如果提供了标量，那么loss只是按给定值缩放.如果weights是一个大小为[batch_size]的张量，
        # 那么批次的每个样本的总损失由weights向量中的相应元素重新调整.如果weights的形状与predictions的形状相匹配，则predictions中每个可测量元素的loss由相应的weights值缩放.
        train_opt = tf.train.RMSPropOptimizer(6.4e-4).minimize(loss=train_loss)  # 初始化一个中RMSprop优化器optimizer之后，通过minimize函数，最小化损失函数

        # 创建会话

        # 启动图，运行op
        with tf.Session() as sess:

            batch_size = num_sample  #即一次训练所抓取的数据样本数量为6。

            # 初始化全局变量
            sess.run(tf.global_variables_initializer())  # 对变量进行初始化，真正的赋值操作
            numepochs =10
            for k in range(numepochs): # range() 函数可创建一个整数列表，一般用在 for 循环中。即k到10.
                for i in range(int(len(x_d) / batch_size)):  # 训练300次。 len 返回列表中的项目数：  ，即i到300
                    train_x = self.x_d[batch_size * i:batch_size * (i + 1), :] #训练x
                    train_y = self.y_d[batch_size * i:batch_size * (i + 1), 0, np.newaxis] #在这一位置增加一个一维
                    feed_data = {x_node: train_x, y_node: train_y} #？
                    _, temp_loss = sess.run([train_opt, train_loss], feed_dict=feed_data)
                    if i % 10 == 0:
                        print(temp_loss)
            Pred = sess.run(y_pred, feed_dict={x_node: self.xx_d})  # 预测数据
            Pred_anti = Pred * (max_price - min_price) + min_price  # 预测数据反归一化
            real_anti = self.y2 * (max_price1 - min_price1) + min_price1  # 真实数据反归一化
            #error = np.mean(np.abs(np.transpose(Pred_anti) - real_anti) / real_anti) #误差
            #MAE = np.mean(np.abs(np.transpose(Pred_anti) - real_anti))
            #RMSE = np.sqrt(np.mean(np.transpose(Pred_anti) - real_anti) ** 2) #R方
        return  Pred_anti, real_anti


def inputdata(j, data1, data2, data_original, data_original1):
    m = len(data1)
    m1 = len(data2)
    data_price = data1.reshape(m, 1) #在不更改数据的情况下为数组赋予新形状。（行，列）
    data_load = data2.reshape(m1, 1)
    Scar = MinMaxScaler()#归一化处理

    # 归一化
    max_price = max(data_original)
    min_price = min(data_original)
    max_price1 = max(data_original1)
    min_price1 = min(data_original1)
    max_load = max(data_load)
    min_load = min(data_load)
    data = (data_price-min_price)/(max_price-min_price)  #价格
    datal = (data_load-min_load)/(max_load-min_load)     #负荷
    data_real = np.array(data_original1).reshape(m, 1)   #真实价格
    data_original_norm = (data_real-min_price1)/(max_price1-min_price1)  #真实价格
    # 制作训练样本% 提前一小时预测

    time_train_start = 24 * 3
    time_train_end = N_T * (90 + j)   # 开始时间
    time_test_begin = time_train_end - N_T  # N_T * (59 + j)
    x1 = datal[time_train_start + N_T:time_train_end + N_T, 0]  # 特征值3
    x2 = data[time_train_start - N_T:time_train_end - N_T, 0]  # 特征值1
    x3 = data[time_train_start - 2 * N_T:time_train_end - 2 * N_T, 0]  # 特征值2
    x4 = data[time_train_start:time_train_end, 0]  # 特征值4
    x5 = data[time_train_start - 1:time_train_end - 1, 0]  # 特征值5
    x6 = data[time_train_start - 2:time_train_end - 2, 0]  # 特征值6
    x7 = data[time_train_start - 3:time_train_end - 3, 0]  # 特征值7
    x8 = data[time_train_start - 4:time_train_end - 4, 0]  # 特征值8
    x9 = datal[time_train_start + N_T - 1:time_train_end + N_T - 1, 0]  # 特征值9

    x_d = np.transpose(np.vstack((x1,x2, x3, x4, x5, x6, x7, x8, x9))) #np.transpose表示维度变换，np.vstack表示按垂直方向（行顺序）堆叠数组构成一个新的数组
    y_d = data[time_train_start + N_T:time_train_end + N_T, 0, np.newaxis]  # 预测界

    #  获取测试样本
    x11 = datal[time_test_begin + 2 * N_T:time_train_end + 2 * N_T, 0]  # 特征值3负荷
    x12 = data[time_test_begin:time_train_end, 0]  # 特征值1
    x13 = data[time_test_begin - N_T: time_train_end - N_T, 0]  # 特征值2
    x14 = data[time_test_begin + N_T:time_train_end + N_T, 0]  # 特征值4
    x15 = data[time_test_begin + N_T - 1:time_train_end + N_T - 1, 0]  # 特征值5
    x16 = data[time_test_begin + N_T - 2:time_train_end + N_T - 2, 0]  # 特征值6
    x17 = data[time_test_begin + N_T - 3:time_train_end + N_T - 3, 0]  # 特征值7
    x18 = data[time_test_begin + N_T - 4:time_train_end + N_T - 4, 0]  # 特征值8
    x19 = datal[time_test_begin + 2 * N_T - 1:time_train_end + 2 * N_T - 1, 0]  # 特征值9

    xx_d = np.transpose(np.vstack((x11, x12, x13, x14, x15, x16, x17, x18, x19)))

    y2 = data_original_norm[time_test_begin + 2 * N_T:time_train_end + 2 * N_T, 0] #对原始数据进行欧几里德范数处理
    return x_d, y_d, xx_d, y2, max_price, min_price, max_price1, min_price1

if __name__ == '__main__':
    # 导入价格数据
    N_T = 24
    #读取数据
    #负荷数据读取：
    df2 = pd.read_csv('负荷小波变换数据.csv')

    # 真实数据集：
    df3 = pd.read_csv('ddf.csv')
    df21= df3.T
    data3 = df21.values[1, :]
    data2 = df21.values[2, :]
    data_original1 = data3

    #电价数据读取：
    df1 = pd.read_csv('小波变换数据.csv')
    dff1 = df1.T

    Num = 30  #预测数量
    ER = np.zeros(Num) # NUM个0 1行7列
    MME = np.zeros(Num)
    RSE = np.zeros(Num)
    SWAPE = np.zeros(Num)
    REAL = np.zeros((Num, N_T))
    PRED = np.zeros((Num, N_T))
    zj = np.zeros(Num*N_T)
    zj1 = np.zeros(Num * N_T)
    ycz = np.zeros((4,Num*N_T))
    PRED1 = np.zeros((Num, N_T))

    for k in  range(len(df1)):
        data1 = df1.values[k, :]
        #data2 = df2.values[k, :]
        data_original = data1
        for j in range(Num):  # j到7
            # 构建训练集和测试集（提前一天、T-24）
            x_d, y_d, xx_d, y2, max_price, min_price, max_price1, min_price1 = \
                inputdata(j, data1, data2, data_original, data_original1)  # inputdata依次输入每个元素的值

            mybp = BPNetwork(x_d, y_d, xx_d, y2, max_price, min_price, max_price1, min_price1)  # BP神经网络
            Pred_anti, real_anti = mybp.BPNN()
            #ER[j] = error
            #MME[j] = MAE
            #RSE[j] = RMSE
            PRED[j, :] = np.transpose(Pred_anti)
            REAL[j, :] = \
                real_anti
        zj = PRED.reshape(Num*N_T)
        ycz[k] = zj
    zj1 = np.sum(ycz,axis=0)
    PRED1 = zj1.reshape(Num, N_T)

    # print(ER, np.mean(ER)) # mean(ER)求取平均值
    PD = pd.DataFrame(PRED1) #创建一个DataFrame
    RD = pd.DataFrame(REAL)
    RRerror_=abs(PD-RD)/RD
    mean_RRerror_ = np.mean(np.transpose(RRerror_))
    print(mean_RRerror_,np.mean(mean_RRerror_))
    ERROR = pd.DataFrame(mean_RRerror_)
    MME = np.mean(np.abs(PD- RD))
    RMM = pd.DataFrame(MME)
    RSE = np.sqrt(np.mean(PD - RD) ** 2)
    RSS = pd.DataFrame(RSE)
    SWAPE = smape(PD,RD)
    SWXPE = pd.DataFrame(SWAPE)
    PD.to_excel('lstm预测数据新.xlsx') #将得到的数据写入excel表格
    RD.to_excel('lstm实际数据新.xlsx')
    ERROR.to_excel('lstmMAPE新.xlsx')
    RMM.to_excel('lstmRMM新.xlsx')  #MAE 平均绝对误差
    RSS.to_excel('lstmRSS新.xlsx')  #MSE 均方绝对误差
    SWXPE.to_excel('lstmSWXPE新.xlsx')  #SMAPE  对称平均绝对百分比误差