#import necessary libraries
from matplotlib import pyplot as plt
import pylab as pl
import numpy as np
# load MNIST data
# start tensorflow interactiveSession
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
CLASS_NUM = 10
K = CLASS_NUM

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
def relu_evidence(logits):
    return tf.nn.relu(logits)
# This one usually works better and used for the second and third examples
# For general settings and different datasets, you may try this one first
def exp_evidence(logits): 
    return tf.exp(tf.clip_by_value(logits,-10,10))
# This one is another alternative and 
# usually behaves better than the relu_evidence 
def softplus_evidence(logits):
    return tf.nn.softplus(logits)
def KL(alpha):
    beta=tf.constant(np.ones((1,10)),dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha,axis=1,keepdims=True)
    S_beta = tf.reduce_sum(beta,axis=1,keepdims=True)
    lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha),axis=1,keepdims=True)
    lnB_uni = tf.reduce_sum(tf.lgamma(beta),axis=1,keepdims=True) - tf.lgamma(S_beta)
    
    dg0 = tf.digamma(S_alpha)
    dg1 = tf.digamma(alpha)
    
    kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
    return kl

def mse_loss(p, alpha, global_step, annealing_step): 
    S = tf.reduce_sum(alpha, axis=1, keepdims=True) 
    E = alpha - 1
    m = alpha / S    
    A = tf.reduce_sum((p-m)**2, axis=1, keepdims=True) 
    B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)     
    annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))   
    alp = E*(1-p) + 1 
    C =  annealing_coef * KL(alp)
    return (A + B) + C

# define some utility functions
def var(name, shape, init=None):
    if init is None:
        init = tf.truncated_normal_initializer(stddev=(2/shape[0])**0.5)
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                          initializer=init)
def CNN_EDL(logits2evidence=relu_evidence,loss_function=mse_loss, lmb=0.001):
    g = tf.Graph()
    with g.as_default():
        # Create the model
        # placeholder
        x = tf.placeholder("float", [None, 1,28,28],name = "in_x")
        y_ = tf.placeholder("float", [None, CLASS_NUM],name = "in_y")
        keep_prob = tf.placeholder("float", name = "ke_prob")
        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        annealing_step = tf.placeholder(dtype=tf.int32)

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
        
        evidence = logits2evidence(logits)
        alpha = evidence + 1
        tf.add_to_collection('evidence', evidence)
       
        u = K / tf.reduce_sum(alpha, axis=1, keepdims=True) #uncertainty #############
       
        prob = alpha/tf.reduce_sum(alpha, 1, keepdims=True) 
        tf.add_to_collection('prob', prob)
       
        loss = tf.reduce_mean(loss_function(y_, alpha, global_step, annealing_step))
        l2_loss = (tf.nn.l2_loss(w_fc1)+tf.nn.l2_loss(w_fc2)) * lmb     #############
      
        step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss + l2_loss, global_step=global_step)
        tf.add_to_collection('prob', prob)
        # Calculate accuracy
        pred = tf.argmax(logits, 1)
        label_p,idx_p,count_p = tf.unique_with_counts(pred)
        tf.add_to_collection('label_p', label_p)
        tf.add_to_collection('idx_p', idx_p)
        tf.add_to_collection('count_p', count_p)
        truth = tf.argmax(y_, 1)
        label,idx,count = tf.unique_with_counts(truth)
        tf.add_to_collection('label', label)
        tf.add_to_collection('idx', idx)
        tf.add_to_collection('count', count)
        
        correct_prediction = tf.equal(pred, truth)
        correct_label=tf.boolean_mask(truth,correct_prediction)
        label_c,idx_c,count_c=tf.unique_with_counts(correct_label) 
        tf.add_to_collection('label_c', label_c)
        tf.add_to_collection('idx_c', idx_c)
        tf.add_to_collection('count_c', count_c)
        
        match = tf.reshape(tf.cast(correct_prediction, tf.float32),(-1,1))
        acc = tf.reduce_mean(match)
       
        total_evidence = tf.reduce_sum(evidence,1, keepdims=True) 
        mean_ev = tf.reduce_mean(total_evidence)
        total_evidence_ture = total_evidence*match
        total_evidence_fail = tf.reduce_sum(evidence,1, keepdims=True)*(1-match)
        total_true = tf.reduce_sum(match)+1e-20
        total_fail = tf.reduce_sum(tf.abs(1-match))+1e-20
        
        
        mean_ev_succ = tf.reduce_sum(total_evidence_ture) / total_true
        mean_ev_fail = tf.reduce_sum(total_evidence_fail) / total_fail 
        
        return g, step, x, y_, annealing_step, keep_prob, prob, acc, loss, u, evidence, mean_ev, \
            mean_ev_succ, mean_ev_fail, label, count, label_p, count_p, label_c, count_c
def draw_EDL_results(file, train_acc1, train_ev_s, train_ev_f, test_acc1, test_ev_s, test_ev_f): 
    # calculate uncertainty for training and testing data for correctly and misclassified samples
    train_u_succ = K / (K+np.array(train_ev_s))
    train_u_fail = K / (K+np.array(train_ev_f))
    test_u_succ  = K / (K+np.array(test_ev_s))
    test_u_fail  = K / (K+np.array(test_ev_f))
    
    f, axs = pl.subplots(2, 2)
    f.set_size_inches([10,10])
    
    axs[0,0].plot(train_ev_s,c='r',marker='+')
    axs[0,0].plot(train_ev_f,c='k',marker='x')
    axs[0,0].set_title('Train Data')
    axs[0,0].set_xlabel('Epoch')
    axs[0,0].set_ylabel('Estimated total evidence for classification') 
    axs[0,0].legend(['Correct Clasifications','Misclasifications'])
    
    
    axs[0,1].plot(train_u_succ,c='r',marker='+')
    axs[0,1].plot(train_u_fail,c='k',marker='x')
    axs[0,1].plot(train_acc1,c='blue',marker='*')
    axs[0,1].set_title('Train Data')
    axs[0,1].set_xlabel('Epoch')
    axs[0,1].set_ylabel('Estimated uncertainty for classification')
    axs[0,1].legend(['Correct clasifications','Misclasifications', 'Accuracy'])
    
    axs[1,0].plot(test_ev_s,c='r',marker='+')
    axs[1,0].plot(test_ev_f,c='k',marker='x')
    axs[1,0].set_title('Test Data')
    axs[1,0].set_xlabel('Epoch')
    axs[1,0].set_ylabel('Estimated total evidence for classification') 
    axs[1,0].legend(['Correct Clasifications','Misclasifications'])
    
    
    axs[1,1].plot(test_u_succ,c='r',marker='+')
    axs[1,1].plot(test_u_fail,c='k',marker='x')
    axs[1,1].plot(test_acc1,c='blue',marker='*')
    axs[1,1].set_title('Test Data')
    axs[1,1].set_xlabel('Epoch')
    axs[1,1].set_ylabel('Estimated uncertainty for classification')
    axs[1,1].legend(['Correct clasifications','Misclasifications', 'Accuracy'])
    
    plt.savefig(file, bbox_inches='tight', dpi=600, format="svg")
    plt.show()

def loss_EDL(func=tf.digamma):
    def loss_func(p, alpha, global_step, annealing_step): 
        S = tf.reduce_sum(alpha, axis=1, keepdims=True) 
        E = alpha - 1
    
        A = tf.reduce_sum(p * (func(S) - func(alpha)), axis=1, keepdims=True)
    
        annealing_coef = tf.minimum(1.0, tf.cast(global_step/annealing_step,tf.float32))
    
        alp = E*(1-p) + 1 
        B =  annealing_coef * KL(alp)
    
        return A + B
    return loss_func
    



        




    

