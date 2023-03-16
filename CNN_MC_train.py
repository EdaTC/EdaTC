import time
import sys
import numpy as np
import os
from tqdm import trange
import csv


# load MNIST data
import input_data
# start tensorflow interactiveSession
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Note: if class numer is 2 or 20, please edit the variable named "num_classes" in /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py"
# DATA_DIR = sys.argv[1]
# CLASS_NUM = int(sys.argv[2])
# EPOCH = int(sys.argv[3])
# GLOBAL_KEEP_B = float(sys.argv[4])

dict_2class = {0:'Benign',1:'Malware'}
dict_10class_benign = {0:'BitTorrent',1:'Facetime',2:'FTP',3:'Gmail',4:'MySQL',5:'Outlook',6:'Skype',7:'SMB',8:'Weibo',9:'WorldOfWarcraft'}
dict_10class_malware = {0:'Cridex',1:'Geodo',2:'Htbot',3:'Miuref',4:'Neris',5:'Nsis-ay',6:'Shifu',7:'Tinba',8:'Virut',9:'Zeus'}
dict_20class = {0:'BitTorrent',1:'Facetime',2:'FTP',3:'Gmail',4:'MySQL',5:'Outlook',6:'Skype',7:'SMB',8:'Weibo',9:'WorldOfWarcraft',10:'Cridex',11:'Geodo',12:'Htbot',13:'Miuref',14:'Neris',15:'Nsis-ay',16:'Shifu',17:'Tinba',18:'Virut',19:'Zeus'}
dict = {}

# function: find a element in a list
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def CNN(keep_prob):
    g = tf.Graph()
    with g.as_default():
        # Create the model
        # placeholder
        x = tf.placeholder("float", [None, 1,28,28],name = "in_x")
        y_ = tf.placeholder("float", [None, CLASS_NUM],name = "in_y")
        keep_prob = tf.placeholder("float", name = "ke_prob")
        global_step = tf.Variable(initial_value=0, name='global_step')

        # first convolutinal layer
        w_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # second convolutional layer
        w_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # densely connected layer
        w_fc1 = weight_variable([7*7*64, 500])
        b_fc1 = bias_variable([500])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        # dropout
        
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

        # readout layer
        w_fc2 = weight_variable([500, CLASS_NUM])
        b_fc2 = bias_variable([CLASS_NUM])

        logits = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
        y_conv = tf.nn.softmax(logits)
        tf.add_to_collection('y_conv', y_conv)

       # define var&op of training&testing
        actual_label = tf.argmax(y_, 1)
        label,idx,count = tf.unique_with_counts(actual_label)
        tf.add_to_collection('label', label)
        tf.add_to_collection('idx', idx)
        tf.add_to_collection('count', count)
        loss = -tf.reduce_sum(y_*tf.log(y_conv))
        train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
        
        predict_label = tf.argmax(y_conv, 1)
        label_p,idx_p,count_p = tf.unique_with_counts(predict_label)
        tf.add_to_collection('label_p', label_p)
        tf.add_to_collection('idx_p', idx_p)
        tf.add_to_collection('count_p', count_p)
        correct_prediction = tf.equal(predict_label, actual_label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.add_to_collection('accuracy', accuracy)

        correct_label=tf.boolean_mask(actual_label,correct_prediction)
        label_c,idx_c,count_c=tf.unique_with_counts(correct_label) 
        tf.add_to_collection('label_c', label_c)
        tf.add_to_collection('idx_c', idx_c)
        tf.add_to_collection('count_c', count_c)
        # Calculate accuracy

        return g, train_step, x, y_, y_conv, keep_prob, accuracy, loss, label, count, label_p, count_p, label_c, count_c

CLASS_NUM = 10
EPOCH = 20
GLOBAL_KEEP_B = 0.5
Data = ["BenignFlowAllLayers","BenignFlowL7","BenignSessionAllLayers","BenignSessionL7","MalwareFlowAllLayers" ,"MalwareFlowL7" ,"MalwareSessionAllLayers" ,"MalwareSessionL7"]
Data1 = ["BenignFlowL7"]
for data_file in Data:
    DATA_DIR = "D:/work/DeepTraffic0428/1.malware_traffic_classification/3.PreprocessedResults/10class/" + data_file
    folder = os.path.split(DATA_DIR)[1]

# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('data_dir', DATA_DIR, 'Directory for storing data')

    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True, num_classes=CLASS_NUM)
    g2, train_step2, X2, Y2, Y_sigmod2, keep_prob2, acc2, loss2, label2,count2, label_p2, count_p2, label_c2, count_c2 = CNN(GLOBAL_KEEP_B)   
    sess2 = tf.Session(graph=g2)
    with g2.as_default(): 
        sess2.run(tf.global_variables_initializer())    
    # if model exists: restore it
    # else: train a new model and save it
        saver = tf.train.Saver()
        model_name = "MC_Dropout_model_" + str(CLASS_NUM) + "class_" + folder + "_" + str(GLOBAL_KEEP_B)
        model =  model_name + '/' + model_name + ".ckpt"
        if not os.path.exists(model):
            if not os.path.exists(model_name):
                os.makedirs(model_name)
        # with open('out.txt','a') as f:
        #     f.write(time.strftime('%Y-%m-%d %X',time.localtime()) + "\n")
        #     f.write('DATA_DIR: ' + DATA_DIR+ "\n")
            bsize = 50
            n_batches = mnist.train.num_examples // bsize
            for epoch in trange(EPOCH,position=0):
                for i in trange(n_batches,position=0):
                    batch = mnist.train.next_batch(bsize)
                    sess2.run(train_step2,feed_dict={X2:batch[0], Y2:batch[1], keep_prob2:GLOBAL_KEEP_B})
                    #print('epoch %d - %d%%) '% (epoch+1, (100*(i+1))//n_batches), end='\r' if i<n_batches-1 else '')
                total_trainbatch, train_accuracy = 0,0
                for _ in trange(50,position=0):
                    batch = mnist.train.next_batch(bsize)
                    train_acc = sess2.run(acc2,feed_dict={X2:batch[0], Y2:batch[1], keep_prob2:GLOBAL_KEEP_B})
                    total_trainbatch += 1
                    train_accuracy += train_acc
                train_accuracy = train_accuracy/total_trainbatch
                test_accuracy=sess2.run(acc2,{X2: mnist.test.images, Y2: mnist.test.labels, keep_prob2:GLOBAL_KEEP_B})
                print('training: %2.4f \t testing: %2.4f' % (train_accuracy, test_accuracy)) 
            save_path = saver.save(sess2, model)
            print("Model saved in file:", save_path)
        else:        
            saver.restore(sess2, model)
            print("Model restored: " + model)
    if CLASS_NUM == 10:
        if folder.startswith('Benign'):
            dict = dict_10class_benign
        elif folder.startswith('Malware'):
            dict = dict_10class_malware
    label_test,count_test,label_p_test,count_p_test,label_c_test,count_c_test,acc_test=sess2.run([label2,count2,label_p2,count_p2,label_c2,count_c2,acc2],{X2: mnist.test.images, Y2: mnist.test.labels, keep_prob2:GLOBAL_KEEP_B})
    
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
    with open('MC_train_accuracy'+str(GLOBAL_KEEP_B)+'.txt','a') as f:
        f.write("\n")
        t = time.strftime('%Y-%m-%d %X',time.localtime())
        f.write(t + "\n")
        f.write('DATA_DIR: ' + DATA_DIR + "\n")
        for item in acc_list:
            f.write(', '.join(item) + "\n")
        f.write('Total accuracy: ' + str(acc_test) + "\n\n")


'''
# new code 
f = open(model_name+'.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
pred_dctlst =np.zeros((len(mnist.test.labels),3)) #{"answer": [], "actual": [], "var": []} #{"answer": [], "actual": [], "var": []}
for i in trange(len(mnist.test.labels)):
    image = mnist.test.images[i]
    actual_label = mnist.test.labels[i].reshape(1,10)
    #print(actual_label)
    print("\n")
    image = image[np.newaxis]
    preds = np.zeros((100, CLASS_NUM), dtype=np.float32)
    predictions = np.zeros(100)
    for j in range(100):
        # Enable dropout
        predictions[j] = predict_label.eval(feed_dict={x:image, y_:actual_label, keep_prob:GLOBAL_KEEP_B})
        #preds[j, :] = predictions
    #print(predictions)
    counts = np.bincount(predictions.astype(int))
    #返回众数
    preds = predictions[np.argmax(counts)]
    #preds = preds.mean(axis=0)
    variance = np.var(predictions,ddof = 1)
    pred_dctlst[i][0] = preds
    pred_dctlst[i][1] = np.argmax(mnist.test.labels[i])
    pred_dctlst[i][2] = variance
    csv_writer.writerow(pred_dctlst[i])
    print(pred_dctlst[i])
f.close()

'''



