# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 16:43:28 2022

@author: 11420
"""

import time
import sys
import numpy as np
import os
from tqdm import trange
import csv
import pandas as pd

# load MNIST data
import input_data
# start tensorflow interactiveSession
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


CLASS_NUM = 10
GLOBAL_KEEP_B = 0.5
Data = ["BenignFlowAllLayers","BenignFlowL7","BenignSessionAllLayers","BenignSessionL7","MalwareFlowAllLayers" ,"MalwareFlowL7" ,"MalwareSessionAllLayers" ,"MalwareSessionL7"]
Data1 = ["MalwareSessionL7"]
for data_file in Data1:
    DATA_DIR = "D:/work/DeepTraffic0428/1.malware_traffic_classification/3.PreprocessedResults/10class/" + data_file
    folder = os.path.split(DATA_DIR)[1]
    for test in ["Benign"]:
        if data_file.endswith('FlowAllLayers'):
            TEST_MODEL_FOLDER = test + 'FlowAllLayers'
        if data_file.endswith('FlowL7'):   
            TEST_MODEL_FOLDER = test + 'FlowL7'
        if data_file.endswith('SessionAllLayers'):
            TEST_MODEL_FOLDER = test + 'SessionAllLayers'
        if data_file.endswith('SessionL7'):   
            TEST_MODEL_FOLDER = test + 'SessionL7'
        
        dict = {}
        folder = os.path.split(DATA_DIR)[1]
        
        sess = tf.compat.v1.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        
        # 模型读取
        model_name = "MC_Dropout_model_" + str(CLASS_NUM) + "class_" + TEST_MODEL_FOLDER + '_' +str(GLOBAL_KEEP_B)
        model_path =  model_name + '/' + model_name
        model = model_name + "/" + model_name + ".ckpt"
        
        saver = tf.train.import_meta_graph('./'+model_path + ".ckpt.meta")# 加载图结构
        saver.restore(sess,'./'+model_path + ".ckpt")
        graph = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
        tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]# 得到当前图中所有变量的名称
        y_conv = tf.get_collection('y_conv')[0]
        accuracy = tf.get_collection('accuracy')[0]
        
        #数据读取
        x = graph.get_operation_by_name('in_x').outputs[0]
        y_= graph.get_operation_by_name('in_y').outputs[0]
        keep_prob = graph.get_operation_by_name('ke_prob').outputs[0]
        print(keep_prob)
        
        test_seat = 100
        
        mnist = input_data.read_data_sets(DATA_DIR, one_hot=True, num_classes=CLASS_NUM)
        
        #f = open(model_name+'_test_'+ folder + '_entropy.csv','a',encoding='utf-8',newline='')
        #csv_writer = csv.writer(f)
        dropout_predictions = np.empty((0,len(mnist.test.labels), CLASS_NUM))  #len(mnist.test.labels)
        softmax = tf.nn.softmax(dropout_predictions,dim=1)
        for i in range(test_seat):
            predictions = np.empty((0, CLASS_NUM))
            for i in trange(len(mnist.test.labels)):
                y_out  = sess.run(y_conv,feed_dict={x:mnist.test.images[i].reshape(1,1,28,28), y_:mnist.test.labels[i].reshape(1,10), keep_prob: GLOBAL_KEEP_B})
                predictions = np.vstack((predictions,y_out))
        
            dropout_predictions = np.vstack((dropout_predictions,predictions[np.newaxis, :, :]))
            # dropout predictions - shape (forward_passes, n_samples, n_classes)
            
        # Calculating mean across multiple MCD forward passes  此处用来计算sofamax的均值
        mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)
        predict = np.argmax(mean)
        # Calculating variance across multiple MCD forward passes 
        
        variance_predict = np.argmax(dropout_predictions,axis=2)    #此处用来计算多次估计的方差
        variance = np.var(variance_predict, axis=0) # shape (n_samples, n_classes)
        
        epsilon = sys.float_info.min
        # Calculating entropy across multiple MCD forward passes 
        entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)
        
        # Calculating mutual information across multiple MCD forward passes 
        mutual_info = entropy - np.mean(np.sum(-dropout_predictions*np.log(dropout_predictions + epsilon),axis=-1), axis=0) # shape (n_samples,)
        print(variance, entropy, mutual_info)
        dataframe = pd.DataFrame({"variance":variance,"entropy":entropy,"mutual_info":mutual_info})
        dataframe.to_csv('./data/'+ model_name+ '_test_'+ folder + '_entropy.csv',index=False,sep=',')
        #f.close()