# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 16:43:28 2022

@author: 11420
"""



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


PROB_HOLD = 1.0
CLASS_NUM = 10
Data = ["BenignFlowAllLayers","BenignFlowL7","BenignSessionAllLayers","BenignSessionL7","MalwareFlowAllLayers" ,"MalwareFlowL7" ,"MalwareSessionAllLayers" ,"MalwareSessionL7"]
Data1 = ["MalwareSessionAllLayers"]
for model_name in Data:
    for test in ["Malware","Benign"]:
    #for test in ["Benign"]:
        if model_name.endswith('FlowAllLayers'):
            test_name = test + 'FlowAllLayers'
        if model_name.endswith('FlowL7'):   
            test_name = test + 'FlowL7'
        if model_name.endswith('SessionAllLayers'):
            test_name = test + 'SessionAllLayers'
        if model_name.endswith('SessionL7'):   
            test_name = test + 'SessionL7'
        test_filename = "D:/work/DeepTraffic0428/1.malware_traffic_classification/3.PreprocessedResults/10class/" + test_name
        mnist = input_data.read_data_sets(test_filename, one_hot=True, num_classes=CLASS_NUM) 
        for method in ["EDL_MSE","EDL_NLEL","EDL_CE"]:
        #for method in ["EDL_MSE"]:
            
            modelfile = 'model_' + method + '_10class_'+ model_name + '_0.5'
            
            sess = tf.compat.v1.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            # 模型读取
            
            model_path =  modelfile + '/' + modelfile
            model = './train_model/' + modelfile + "/" + modelfile + ".ckpt"
            
            saver = tf.train.import_meta_graph(model + ".meta")# 加载图结构
            saver.restore(sess, model)
            graph = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
            tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]# 得到当前图中所有变量的名称
            
            #数据读取
            x = graph.get_operation_by_name('in_x').outputs[0]
            y_= graph.get_operation_by_name('in_y').outputs[0]
            keep_prob = graph.get_operation_by_name('ke_prob').outputs[0]
            
            evidence = tf.get_collection('evidence')[0]
            prob = tf.get_collection('prob')[0]
            # label_p = graph.get_operation_by_name('label_p').outputs[0]
            # idx_p = graph.get_operation_by_name('idx_p').outputs[0]
            # count_p = graph.get_operation_by_name('count_p').outputs[0]
            # label = graph.get_operation_by_name('label').outputs[0]
            # idx = graph.get_operation_by_name('idx').outputs[0]
            # count = graph.get_operation_by_name('count').outputs[0]
            # label_c = graph.get_operation_by_name('label_c').outputs[0]
            # idx_c = graph.get_operation_by_name('idx_c').outputs[0]
            # count_c = graph.get_operation_by_name('count_c').outputs[0]
            
            #   测试数据读取
            
            
            pred_1, prd_2, pred_3 =[],[],[]
            
            test_num = 10
            
            #un_evidence
            #test_av_evidence = np.empty((0,len(mnist.test.labels), 100)) 
            test_prob,test_evidence = sess.run([prob,evidence], feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:PROB_HOLD})
            pre_y = np.argmax(test_prob,axis=1).tolist()
            prob = np.amax(test_prob,axis=1).tolist()
            act_y = np.argmax(mnist.test.labels,axis=1).tolist()
            pred_uncertainty = (10 / tf.reduce_sum(test_evidence+1, axis=1, keepdims=True))
            with tf.Session() as sess: 
                pred_uncertainty = pred_uncertainty.eval()
            pred_uncertainty = pred_uncertainty.reshape(pred_uncertainty.shape[0])
            
            dataframe = pd.DataFrame({"pre_y":pre_y,"act_y":act_y,"pred_uncertainty":pred_uncertainty,'prob':prob})
            dataframe.to_csv('model_'+ model_name+ '_test_'+test_name + "_method_" + method + '.csv',index=False,sep=',')
            print('./test_evidence_data0.5/model_'+ model_name+ '_test_'+test_name)
            

            

