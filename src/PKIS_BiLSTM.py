# -*- coding: utf-8 -*-

import tensorflow as tf
import uuid
import tensorflow.contrib as contrib
 

class bilstm(object):
    def __init__(self, in_data, hidden_dim, batch_seqlen=None, flag='concat'):
        self.in_data = in_data
        self.hidden_dim = hidden_dim
        self.batch_seqlen = batch_seqlen
        self.flag = flag
        
        print("**********************************")
        # tf.reset_default_graph() 
        lstm_cell_fw = contrib.rnn.LSTMCell(self.hidden_dim, name= str(uuid.uuid1()))
        lstm_cell_bw = contrib.rnn.LSTMCell(self.hidden_dim, name= str(uuid.uuid1()))
        out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,cell_bw=lstm_cell_bw,inputs=self.in_data,dtype=tf.float32)
        bi_out = tf.concat(out, 2)
        if flag=='all_ht':
            self.out = bi_out
        if flag=='first_ht':
            self.out = bi_out[:,0,:]
        if flag=='last_ht':
            self.out = tf.concat([state[0].h,state[1].h], 1)
        if flag=='concat':
            self.out = tf.concat([bi_out[:,0,:],tf.concat([state[0].h,state[1].h], 1)],1)
    
    def return_bilstm(self):
        return self.out








