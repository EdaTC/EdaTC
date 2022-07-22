# -*- coding: utf-8 -*-
"""
Created on Thu May 19 09:50:35 2022

@author: 11420
"""



import pandas as pd
import time
import os
from tqdm import trange
from input_data import read_data_sets
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from cnn_evidence import CNN_EDL,draw_EDL_results,find_element_in_list,exp_evidence, loss_EDL,relu_evidence,mse_loss


dict_2class = {0:'Benign',1:'Malware'}
dict_10class_benign = {0:'BitTorrent',1:'Facetime',2:'FTP',3:'Gmail',4:'MySQL',5:'Outlook',6:'Skype',7:'SMB',8:'Weibo',9:'WorldOfWarcraft'}
dict_10class_malware = {0:'Cridex',1:'Geodo',2:'Htbot',3:'Miuref',4:'Neris',5:'Nsis     -ay',6:'Shifu',7:'Tinba',8:'Virut',9:'Zeus'}
dict_20class = {0:'BitTorrent',1:'Facetime',2:'FTP',3:'Gmail',4:'MySQL',5:'Outlook',6:'Skype',7:'SMB',8:'Weibo',9:'WorldOfWarcraft',10:'Cridex',11:'Geodo',12:'Htbot',13:'Miuref',14:'Neris',15:'Nsis-ay',16:'Shifu',17:'Tinba',18:'Virut',19:'Zeus'}
dict = {}

CLASS_NUM = 10
EPOCH = 20
GLOBAL_KEEP_B_ = [0.5]
TEST_KEEP_B = 1.0
K = CLASS_NUM


            
Data = ["BenignFlowAllLayers","BenignFlowL7","BenignSessionAllLayers","BenignSessionL7","MalwareFlowAllLayers" ,"MalwareFlowL7" ,"MalwareSessionAllLayers" ,"MalwareSessionL7"]
Data1= ["BenignFlowL7"]




  # MSE    5
for GLOBAL_KEEP_B in GLOBAL_KEEP_B_:
    for data_file in Data:
        DATA_DIR = "D:/work/DeepTraffic0428/1.malware_traffic_classification/3.PreprocessedResults/10class/" + data_file
        folder = os.path.split(DATA_DIR)[1]
        mnist = read_data_sets(DATA_DIR, one_hot=True, num_classes=CLASS_NUM)
        if CLASS_NUM == 10:
            if folder.startswith('Benign'):
                dict = dict_10class_benign
            elif folder.startswith('Malware'):
                dict = dict_10class_malware      
        
        g2, step2, X2, Y2, annealing_step2, keep_prob2, prob2, acc2, loss2, u2, evidence2, \
            mean_ev2, mean_ev_succ2, mean_ev_fail2, label2, count2, label_p2, count_p2,\
                label_c2, count_c2 =  CNN_EDL(lmb=0.005)    
        sess2 = tf.Session(graph=g2)
        with g2.as_default():
            sess2.run(tf.global_variables_initializer())   
            saver = tf.train.Saver()
            model_name = "model_EDL_MSE_" + str(CLASS_NUM) + "class_" + folder + "_" + str(GLOBAL_KEEP_B)        
            model =  model_name + '/' + model_name + ".ckpt"
            step_bsize = 50 #batch size
            train_bsize = 10000 #batch size
            test_bsize = 1000 #batch size
            train_n_batches = mnist.train.num_examples // step_bsize
            train_n_batches1 = mnist.train.num_examples // train_bsize
            test_n_batches = mnist.test.num_examples // test_bsize
            
            L_train_acc,L_train_ev_s,L_train_ev_f,L_test_acc,L_test_ev_s, L_test_ev_f=[],[],[],[],[],[]
            if not os.path.exists(model):
                if not os.path.exists(model_name):
                    os.makedirs(model_name)    
          #batch size
                for epoch in trange(EPOCH):   
                    for i in trange(train_n_batches,position=0):
                        data, label = mnist.train.next_batch(step_bsize)
                        feed_dict={X2:data, Y2:label, keep_prob2:GLOBAL_KEEP_B, annealing_step2:10*train_n_batches}
                        sess2.run(step2,feed_dict)
                    train_acc2, train_succ2, train_fail2,total_trainbatch_cnt2 = 0,0,0,0
                    for _ in trange(train_n_batches1,position=0):
                        data, label = mnist.train.next_batch(train_bsize)
                        train_acc, train_succ, train_fail = sess2.run([acc2,mean_ev_succ2,mean_ev_fail2], feed_dict={X2:data,Y2:label,keep_prob2:TEST_KEEP_B})
                        total_trainbatch_cnt2 += 1
                        train_acc2 += train_acc
                        train_succ2 += train_succ
                        train_fail2 += train_fail
                    train_acc = train_acc2/total_trainbatch_cnt2
                    train_succ = train_succ2/total_trainbatch_cnt2
                    train_fail = train_fail2/total_trainbatch_cnt2
                    test_acc, test_succ, test_fail = sess2.run([acc2,mean_ev_succ2,mean_ev_fail2], feed_dict={X2:mnist.test.images,Y2:mnist.test.labels,keep_prob2:TEST_KEEP_B})
                    L_train_acc.append(train_acc)
                    L_train_ev_s.append(train_succ)
                    L_train_ev_f.append(train_fail)
                    L_test_acc.append(test_acc)
                    L_test_ev_s.append(test_succ)
                    L_test_ev_f.append(test_fail)
                    print('training: %2.4f (%2.4f - %2.4f) \t testing: %2.4f (%2.4f - %2.4f)' % (train_acc, train_succ, train_fail, test_acc, test_succ, test_fail))
                dataframe = pd.DataFrame({"L_train_acc":L_train_acc,"L_train_ev_s":L_train_ev_s,"L_train_ev_f":L_train_ev_f,\
                                          "L_test_acc":L_test_acc,"L_test_ev_s":L_test_ev_s,"L_test_ev_f":L_test_ev_f})
                dataframe.to_csv('./data/'+ model_name+ '.csv',index=False,sep=',')
                save_path = saver.save(sess2, model)
                print("Model saved in file:", save_path)
            else:        
                saver.restore(sess2, model)
                print("Model restored: " + model)
            label_test,count_test,label_p_test,count_p_test,label_c_test,count_c_test,acc_test=sess2.run([label2,count2,label_p2,count_p2,label_c2,count_c2,acc2],{X2: mnist.test.images, Y2: mnist.test.labels, keep_prob2:TEST_KEEP_B})
                
            acc_list = []
            for i in range(CLASS_NUM):
                n1 = find_element_in_list(i,label_test.tolist())
                count_actual = count_test[n1]
                n2 = find_element_in_list(i,label_c_test.tolist())
                count_correct = count_c_test[n2] if n2>-1 else 0
                n3 = find_element_in_list(i,label_p_test.tolist())
                count_predict = count_p_test[n3] if n3>-1 else 0
        
                recall = float(count_correct)/float(count_actual)
                precision = float(count_correct)/float(count_predict) if count_predict>0 else -1
                acc_list.append([str(i),dict[i],str(precision),str(recall)])
            with open("./accuracy/model_EDL_MSE_" + str(CLASS_NUM) + "class_" + folder + "_" + str(GLOBAL_KEEP_B)+'.txt','a') as f:
                f.write("\n")
                t = time.strftime('%Y-%m-%d %X',time.localtime())
                f.write(t + "\n")
                f.write('DATA_DIR: ' + DATA_DIR + "\n")
                for item in acc_list:
                    f.write(', '.join(item) + "\n")
                f.write('Total accuracy: ' + str(acc_test) + "\n\n")            
        draw_EDL_results("./fig/model_EDL_MSE_" + str(CLASS_NUM) + "class_" + folder + "_" + str(GLOBAL_KEEP_B)+'.png',L_train_acc, L_train_ev_s, L_train_ev_f, L_test_acc, L_test_ev_s, L_test_ev_f)
            
    
    
    
    #  CE   4
    
    for data_file in Data:
        DATA_DIR = "D:/work/DeepTraffic0428/1.malware_traffic_classification/3.PreprocessedResults/10class/" + data_file
        folder = os.path.split(DATA_DIR)[1]
        mnist = read_data_sets(DATA_DIR, one_hot=True, num_classes=CLASS_NUM)
        if CLASS_NUM == 10:
            if folder.startswith('Benign'):
                dict = dict_10class_benign
            elif folder.startswith('Malware'):
                dict = dict_10class_malware      
        
            
        #Using the Expected Cross Entropy (Eq. 4)
        g2, step2, X2, Y2, annealing_step2, keep_prob2, prob2, acc2, loss2, u2, evidence2, \
            mean_ev2, mean_ev_succ2, mean_ev_fail2, label2, count2, label_p2, count_p2,\
                label_c2, count_c2 =  CNN_EDL(exp_evidence, loss_EDL(tf.digamma), lmb=0.005)    
        sess2 = tf.Session(graph=g2)
        with g2.as_default():
            sess2.run(tf.global_variables_initializer())   
            saver = tf.train.Saver()
            model_name = "model_EDL_CE_" + str(CLASS_NUM) + "class_" + folder + "_" + str(GLOBAL_KEEP_B)        
            model =  model_name + '/' + model_name + ".ckpt"
            step_bsize = 50 #batch size
            train_bsize = 10000 #batch size
            test_bsize = 1000 #batch size
            train_n_batches = mnist.train.num_examples // step_bsize
            train_n_batches1 = mnist.train.num_examples // train_bsize
            test_n_batches = mnist.test.num_examples // test_bsize
            
            L_train_acc,L_train_ev_s,L_train_ev_f,L_test_acc,L_test_ev_s, L_test_ev_f=[],[],[],[],[],[]
            if not os.path.exists(model):
                if not os.path.exists(model_name):
                    os.makedirs(model_name)    
          #batch size
                for epoch in trange(EPOCH):   
                    for i in trange(train_n_batches,position=0):
                        data, label = mnist.train.next_batch(step_bsize)
                        feed_dict={X2:data, Y2:label, keep_prob2:GLOBAL_KEEP_B, annealing_step2:10*train_n_batches}
                        sess2.run(step2,feed_dict)
                    train_acc2, train_succ2, train_fail2,total_trainbatch_cnt2 = 0,0,0,0
                    for _ in trange(train_n_batches1,position=0):
                        data, label = mnist.train.next_batch(train_bsize)
                        train_acc, train_succ, train_fail = sess2.run([acc2,mean_ev_succ2,mean_ev_fail2], feed_dict={X2:data,Y2:label,keep_prob2:TEST_KEEP_B})
                        total_trainbatch_cnt2 += 1
                        train_acc2 += train_acc
                        train_succ2 += train_succ
                        train_fail2 += train_fail
                    train_acc = train_acc2/total_trainbatch_cnt2
                    train_succ = train_succ2/total_trainbatch_cnt2
                    train_fail = train_fail2/total_trainbatch_cnt2
                    test_acc, test_succ, test_fail = sess2.run([acc2,mean_ev_succ2,mean_ev_fail2], feed_dict={X2:mnist.test.images,Y2:mnist.test.labels,keep_prob2:TEST_KEEP_B})
                    L_train_acc.append(train_acc)
                    L_train_ev_s.append(train_succ)
                    L_train_ev_f.append(train_fail)
                    L_test_acc.append(test_acc)
                    L_test_ev_s.append(test_succ)
                    L_test_ev_f.append(test_fail)
                    print('training: %2.4f (%2.4f - %2.4f) \t testing: %2.4f (%2.4f - %2.4f)' % (train_acc, train_succ, train_fail, test_acc, test_succ, test_fail))
                dataframe = pd.DataFrame({"L_train_acc":L_train_acc,"L_train_ev_s":L_train_ev_s,"L_train_ev_f":L_train_ev_f,\
                                          "L_test_acc":L_test_acc,"L_test_ev_s":L_test_ev_s,"L_test_ev_f":L_test_ev_f})
                dataframe.to_csv('./data/'+ model_name+ '.csv',index=False,sep=',')
                save_path = saver.save(sess2, model)
                print("Model saved in file:", save_path)
            else:        
                saver.restore(sess2, model)
                print("Model restored: " + model)
            label_test,count_test,label_p_test,count_p_test,label_c_test,count_c_test,acc_test=sess2.run([label2,count2,label_p2,count_p2,label_c2,count_c2,acc2],{X2: mnist.test.images, Y2: mnist.test.labels, keep_prob2:TEST_KEEP_B})
                
            acc_list = []
            for i in range(CLASS_NUM):
                n1 = find_element_in_list(i,label_test.tolist())
                count_actual = count_test[n1]
                n2 = find_element_in_list(i,label_c_test.tolist())
                count_correct = count_c_test[n2] if n2>-1 else 0
                n3 = find_element_in_list(i,label_p_test.tolist())
                count_predict = count_p_test[n3] if n3>-1 else 0
        
                recall = float(count_correct)/float(count_actual)
                precision = float(count_correct)/float(count_predict) if count_predict>0 else -1
                acc_list.append([str(i),dict[i],str(precision),str(recall)])
            with open("./accuracy/model_EDL_CE_" + str(CLASS_NUM) + "class_" + folder + "_" + str(GLOBAL_KEEP_B)+'.txt','a') as f:
                f.write("\n")
                t = time.strftime('%Y-%m-%d %X',time.localtime())
                f.write(t + "\n")
                f.write('DATA_DIR: ' + DATA_DIR + "\n")
                for item in acc_list:
                    f.write(', '.join(item) + "\n")
                f.write('Total accuracy: ' + str(acc_test) + "\n\n")            
        draw_EDL_results("./fig/model_EDL_CE_" + str(CLASS_NUM) + "class_" + folder + "_" + str(GLOBAL_KEEP_B)+'.png',L_train_acc, L_train_ev_s, L_train_ev_f, L_test_acc, L_test_ev_s, L_test_ev_f)
            
    # Using Negative Log of the Expected Likelihood   3
            
    for data_file in Data:
        DATA_DIR = "D:/work/DeepTraffic0428/1.malware_traffic_classification/3.PreprocessedResults/10class/" + data_file
        folder = os.path.split(DATA_DIR)[1]
        mnist = read_data_sets(DATA_DIR, one_hot=True, num_classes=CLASS_NUM)
        if CLASS_NUM == 10:
            if folder.startswith('Benign'):
                dict = dict_10class_benign
            elif folder.startswith('Malware'):
                dict = dict_10class_malware
    
        
        #Using Negative Log of the Expected Likelihood 
        
        g2, step2, X2, Y2, annealing_step, keep_prob2, prob2, acc2, loss2, u, evidence, \
            mean_ev, mean_ev_succ, mean_ev_fail, label2, count2, label_p2, count_p2,\
                label_c2, count_c2 = CNN_EDL(exp_evidence, loss_EDL(tf.log), lmb=0.005)
        sess2 = tf.Session(graph=g2)
        with g2.as_default():
            sess2.run(tf.global_variables_initializer())   
            saver = tf.train.Saver()
            model_name = "model_EDL_NLEL_" + str(CLASS_NUM) + "class_" + folder + "_" + str(GLOBAL_KEEP_B)
            model =  model_name + '/' + model_name + ".ckpt"
            step_bsize = 50 #batch size
            train_bsize = 10000 #batch size
            test_bsize = 1000 #batch size
            train_n_batches = mnist.train.num_examples // step_bsize
            train_n_batches1 = mnist.train.num_examples // train_bsize
            test_n_batches = mnist.test.num_examples // test_bsize
            
            L_train_acc,L_train_ev_s,L_train_ev_f,L_test_acc,L_test_ev_s, L_test_ev_f=[],[],[],[],[],[]
            if not os.path.exists(model):
                if not os.path.exists(model_name):
                    os.makedirs(model_name)    
          #batch size
                for epoch in trange(EPOCH):   
                    for i in trange(train_n_batches,position=0):
                        data, label = mnist.train.next_batch(step_bsize)
                        feed_dict={X2:data, Y2:label, keep_prob2:GLOBAL_KEEP_B, annealing_step:10*train_n_batches}
                        sess2.run(step2,feed_dict)
                    train_acc2, train_succ2, train_fail2,total_trainbatch_cnt2 = 0,0,0,0
                    for _ in trange(train_n_batches1,position=0):
                        data, label = mnist.train.next_batch(train_bsize)
                        train_acc, train_succ, train_fail = sess2.run([acc2,mean_ev_succ,mean_ev_fail], feed_dict={X2:data,Y2:label,keep_prob2:TEST_KEEP_B})
                        total_trainbatch_cnt2 += 1
                        train_acc2 += train_acc
                        train_succ2 += train_succ
                        train_fail2 += train_fail
                    train_acc = train_acc2/total_trainbatch_cnt2
                    train_succ = train_succ2/total_trainbatch_cnt2
                    train_fail = train_fail2/total_trainbatch_cnt2
                    test_acc, test_succ, test_fail = sess2.run([acc2,mean_ev_succ,mean_ev_fail], feed_dict={X2:mnist.test.images,Y2:mnist.test.labels,keep_prob2:TEST_KEEP_B})
                    
                    L_train_acc.append(train_acc)
                    L_train_ev_s.append(train_succ)
                    L_train_ev_f.append(train_fail)
                    L_test_acc.append(test_acc)
                    L_test_ev_s.append(test_succ)
                    L_test_ev_f.append(test_fail)
                    print('training: %2.4f (%2.4f - %2.4f) \t testing: %2.4f (%2.4f - %2.4f)' % (train_acc, train_succ, train_fail, test_acc, test_succ, test_fail))
                dataframe = pd.DataFrame({"L_train_acc":L_train_acc,"L_train_ev_s":L_train_ev_s,"L_train_ev_f":L_train_ev_f,\
                                          "L_test_acc":L_test_acc,"L_test_ev_s":L_test_ev_s,"L_test_ev_f":L_test_ev_f})
                dataframe.to_csv('./data/'+ model_name+ '.csv',index=False,sep=',')
                save_path = saver.save(sess2, model)
                print("Model saved in file:", save_path)
            else:        
                saver.restore(sess2, model)
                print("Model restored: " + model)
            label_test,count_test,label_p_test,count_p_test,label_c_test,count_c_test,acc_test=sess2.run([label2,count2,label_p2,count_p2,label_c2,count_c2,acc2],{X2: mnist.test.images, Y2: mnist.test.labels, keep_prob2:TEST_KEEP_B})
                
            acc_list = []
            for i in range(CLASS_NUM):
                n1 = find_element_in_list(i,label_test.tolist())
                count_actual = count_test[n1]
                n2 = find_element_in_list(i,label_c_test.tolist())
                count_correct = count_c_test[n2] if n2>-1 else 0
                n3 = find_element_in_list(i,label_p_test.tolist())
                count_predict = count_p_test[n3] if n3>-1 else 0
        
                recall = float(count_correct)/float(count_actual)
                precision = float(count_correct)/float(count_predict) if count_predict>0 else -1
                acc_list.append([str(i),dict[i],str(precision),str(recall)])
            with open("./accuracy/model_EDL_NLEL_" + str(CLASS_NUM) + "class_" + folder + "_" + str(GLOBAL_KEEP_B)+'.txt','a') as f:
                f.write("\n")
                t = time.strftime('%Y-%m-%d %X',time.localtime())
                f.write(t + "\n")
                f.write('DATA_DIR: ' + DATA_DIR + "\n")
                for item in acc_list:
                    f.write(', '.join(item) + "\n")
                f.write('Total accuracy: ' + str(acc_test) + "\n\n")        
        draw_EDL_results("./fig/model_EDL_NLEL_" + str(CLASS_NUM) + "class_" + folder + "_" + str(GLOBAL_KEEP_B)+".png",L_train_acc, L_train_ev_s, L_train_ev_f, L_test_acc, L_test_ev_s, L_test_ev_f)