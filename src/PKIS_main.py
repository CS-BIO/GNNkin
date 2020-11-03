# -*- coding: utf-8 -*-

"""
Spyder Editor
"""
import os
import argparse
import tensorflow as tf
import numpy as np
import random,math,time
from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve
from gensim.models import Word2Vec
from PKIS_BiLSTM import bilstm

tf.reset_default_graph()

def parse_args():
    parser = argparse.ArgumentParser(description="Run heer.")
    parser.add_argument('--learning_rate', type=float, default=0.005,
	                    help='Learning Rate. Default is 0.005.')
    parser.add_argument('--total_iter', type=int, default=800,
                    	help='total_iter')
    parser.add_argument('--Net_dim', type=int, default=64,
                    	help='Network feature dimensions')
    parser.add_argument('--Seq_dim', type=int, default=128,
                    	help='structure feature dimensions')
    parser.add_argument('--SegSeq_length', type=int, default=1200,
                    	help='structure feature dimensions')
    parser.add_argument('--alpha', type=int, default=0.2,
                    	help='alpha')    
    parser.add_argument('--com_number', type=int, default=366,
                    	help='total_iter')
    parser.add_argument('--kin_number', type=int, default=195,
                    	help='total_iter')
    return parser.parse_args()

def model_train(args, data_cK,Kin_pre,data_cPre,test0,test1,MD_matOrigin):
    xs = tf.placeholder(tf.float32, [args.com_number, 1024]) # 600*600
    ys = tf.placeholder(tf.float32, [args.kin_number, args.SegSeq_length, 64])
    Yture = tf.placeholder(tf.float32,[args.com_number, args.kin_number])
    
    Xnet = tf.Variable(tf.truncated_normal([args.com_number, args.Net_dim],stddev = 0.1,dtype=tf.float32))
    Ynet = tf.Variable(tf.truncated_normal([args.kin_number, args.Net_dim],stddev = 0.1,dtype=tf.float32))
    
    W0 = tf.Variable(tf.truncated_normal([1024, args.Seq_dim],stddev = 0.1,dtype=tf.float32))
    
    W_encoder0 = tf.Variable(tf.truncated_normal([args.Seq_dim, args.Seq_dim],stddev = 0.1,dtype=tf.float32))
    W_encoder1 = tf.Variable(tf.truncated_normal([args.Seq_dim, args.Seq_dim],stddev = 0.1,dtype=tf.float32))
    W1 = tf.Variable(tf.truncated_normal([args.Seq_dim, args.Seq_dim],stddev = 0.1,dtype=tf.float32))
    W2 = tf.Variable(tf.truncated_normal([args.Seq_dim, args.Seq_dim],stddev = 0.1,dtype=tf.float32))
    W3 = tf.Variable(tf.truncated_normal([args.Seq_dim, args.Seq_dim],stddev = 0.1,dtype=tf.float32))
    W4 = tf.Variable(tf.truncated_normal([args.Seq_dim, args.Seq_dim],stddev = 0.1,dtype=tf.float32))
    
    Wx1_IDMP = tf.Variable(tf.truncated_normal([args.Net_dim, args.Net_dim],stddev = 0.1,dtype=tf.float32))
    Wy1_IDMP = tf.Variable(tf.truncated_normal([args.Net_dim, args.Net_dim],stddev = 0.1,dtype=tf.float32))
    Wx2_IDMP = tf.Variable(tf.truncated_normal([args.Net_dim, args.Net_dim],stddev = 0.1,dtype=tf.float32))
    Wy2_IDMP = tf.Variable(tf.truncated_normal([args.Net_dim, args.Net_dim],stddev = 0.1,dtype=tf.float32))
    
    Qm = tf.Variable(tf.truncated_normal([args.Net_dim, args.Seq_dim],stddev = 0.1,dtype=tf.float32))
    Qk = tf.Variable(tf.truncated_normal([args.Net_dim, args.Seq_dim],stddev = 0.1,dtype=tf.float32))

    Wy1 = tf.Variable(tf.truncated_normal([2*args.Seq_dim, args.Seq_dim],stddev = 0.1,dtype=tf.float32))
    B_encoder0 = tf.Variable(tf.truncated_normal([MD_matOrigin.shape[0],args.Seq_dim],stddev = 0.1,dtype=tf.float32))
    B_encoder1 = tf.Variable(tf.truncated_normal([MD_matOrigin.shape[0],args.Seq_dim],stddev = 0.1,dtype=tf.float32))
 
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W_encoder0))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W_encoder1))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W1))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W2))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W3))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W4))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(Wx1_IDMP))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(Wy1_IDMP))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(Wx2_IDMP))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(Wy2_IDMP))
    
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(Qm))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(Qk))
       
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(Wy1))

    xs1 = tf.nn.relu(tf.matmul(xs,W0))
    xs_a = tf.sigmoid(tf.add(tf.matmul(xs1,W_encoder0),B_encoder0)) 
    xs_b = tf.nn.relu(tf.add(tf.matmul(xs1,W_encoder1),B_encoder1))
    matr_Y = tf.nn.softmax(tf.matmul(xs_a,tf.transpose(xs_a)))    

    diag_Y = tf.math.reduce_sum(matr_Y,axis = 0)
    D1 = tf.matrix_diag(tf.pow(diag_Y,-0.5))
    aa = tf.matmul(tf.matmul(D1,matr_Y),D1)

    Xseq1 = tf.sigmoid(tf.matmul(tf.matmul(aa, xs_b), W1))
    Xseq = tf.sigmoid(tf.matmul(tf.matmul(aa, Xseq1), W2)) 
  
    clf = bilstm(ys,0.5*args.Seq_dim)
    ys2 = bilstm.return_bilstm(clf)
    
    Yseq = tf.matmul(ys2,Wy1)
    
    Hc = tf.nn.relu(tf.matmul(tf.matmul(Yture,Ynet),Wx1_IDMP))
    Hk = tf.nn.relu(tf.matmul(tf.matmul(tf.transpose(Yture),Xnet),Wy1_IDMP))
    
    X = tf.add((1-args.alpha)*Xseq, tf.matmul(args.alpha*Hc,Qm))
    Y = tf.add((1-args.alpha)*Yseq, tf.matmul(args.alpha*Hk,Qk))
    
    Ypre = tf.matmul(X,tf.transpose(Y))

    loss = tf.reduce_sum(tf.square(tf.subtract(Yture,Ypre)))

    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
      
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        list_train_loss = []
        for num_iter in range(args.total_iter):
            tic = time.time()
            y_=MD_matOrigin[test0,test1]
            sess.run(optimizer,feed_dict={xs:data_cPre,ys:Kin_pre,Yture:data_cK})  
            train_loss=sess.run(loss,feed_dict={xs:data_cPre,ys:Kin_pre,Yture:data_cK})
            training_time = time.time() - tic
            list_train_loss.append(train_loss)
            reconstruct_v = sess.run(Ypre,feed_dict={xs:data_cPre,ys:Kin_pre,Yture:data_cK})            
            y_pred = reconstruct_v[test0,test1]
                        
            auc_score = roc_auc_score(y_,y_pred)
            fpr, tpr, thresholds = roc_curve(y_, y_pred)
            prec, recall, thresholds = precision_recall_curve(y_, y_pred)
            aupr_score = np.trapz(recall, prec)
            if(num_iter%5==0):
                print('[TRN] iter = %03i, AUC=%.5g, AUPR=%.5g, (%3.2es)'% \
                      (num_iter, auc_score, aupr_score, training_time))

def read_data(path):
    data = []
    for line in open(path, 'r'):
        ele = line.strip().split(" ")
        tmp = []
        for e in ele:
            if e != '':
                tmp.append(float(e))
        data.append(tmp)
    return data

def read_data2(path):
    data = []
    for line in open(path, 'r'):
        ele = line.strip().split(" ")
        tmp = []
        for e in ele:
            if e != '':
                tmp.append(e)
        data.append(tmp)
    return data

if __name__ == '__main__':
    args = parse_args()
    print(args.learning_rate)
    data_path = os.path.join(os.path.dirname(os.getcwd()),"data\PKIS")
    
    data_CompKin = read_data(data_path + '\PISK_Matrix_Compound_Kinase.txt')
    data_CompPre = read_data(data_path + '\PIKS_Compound_feature1024.txt')
    data_KinSeq = read_data2(data_path + '\PIKS_KinSeq.txt')
    
    data_cK = np.array(data_CompKin,dtype = "int8")
    data_cPre = np.array(data_CompPre)
    data_KinSeq = np.array(data_KinSeq)
    
    amino_acid = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    model = Word2Vec.load(data_path + "\word2vec_64.model")
    vector = model.wv["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    
    Kin_pre = []
    for i in range(len(data_KinSeq)): 
        arr_vec = np.zeros((1200,vector.shape[1]))
        for j in range(1200):
            if len(data_KinSeq[i]) > 1200:
                for a in range(len(amino_acid)):
                    if data_KinSeq[i][j] == amino_acid[a]:
                        arr_vec[j,] = vector[a]
            elif len(data_KinSeq[i]) <= 1200:
                if j < len(data_KinSeq[i]):
                    for a in range(len(amino_acid)):
                        if data_KinSeq[i][j] == amino_acid[a]:
                            arr_vec[j,] = vector[a]
                if j >= len(data_KinSeq[i]):
                    arr_vec[j,] = [0]*64
        Kin_pre.append(arr_vec)
    
    Kin_pre = np.array(Kin_pre)
    
    pos_position = np.zeros(shape=(int(np.sum(data_cK)),2))
    tmp_pos = 0
    tmp_arr = []
    for a in range(data_cK.shape[0]):
        for b in range(data_cK.shape[1]):
            if data_cK[a,b] == 1:
                pos_position[tmp_pos,0] = a
                pos_position[tmp_pos,1] = b
                tmp_arr.append(tmp_pos)
                tmp_pos +=1
                

    random.shuffle(tmp_arr)  
    tep_pos_set = tmp_arr
    num_tep = math.floor(len(tep_pos_set)*0.2)
    t = 1
    
    arrAUC = []
    arrAUPR = []
    arrMSE = []

    flag_auc = 0
    for x in range(t):
        data_cK_new = np.zeros(shape = data_cK.shape)
        for i in range(data_cK.shape[0]):
            for j in range(data_cK.shape[1]):
                data_cK_new[i,j] = data_cK[i,j]      
        
        for j in range((x*num_tep),((x+1)*num_tep)):
            data_cK_new[int(pos_position[tep_pos_set[j],0]),int(pos_position[tep_pos_set[j],1])] = 0 
        
        test0 = []
        test1 = []
        for a in range(data_cK_new.shape[0]):
            for b in range(data_cK_new.shape[1]):
                if data_cK_new[a,b] == 0:
                    test0.append(a)
                    test1.append(b)

        model_train(args,data_cK_new,Kin_pre,data_cPre,test0,test1,data_cK)

























