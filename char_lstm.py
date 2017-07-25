import numpy as np
import time
import tensorflow as tf
from keras.utils.data_utils import get_file

path = get_file('nietzsche.txt',
                origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = list(open(path).read())
chars = ['\0']+ list(set(text))
print('Text length:', len(text))
print("Num chars:", len(chars))

char_idx = {a:i for i,a in enumerate(chars)}
idx_char = {i:a for i,a in enumerate(chars)}

seq_len = 40
emb_size = len(chars)
b_size = 64

text = ['\0']*(seq_len-1) + text

X_train = [[char_idx[text[i+j]] for j in range(seq_len)] for i in range(len(text)-seq_len-1)]
Y_train = [[char_idx[text[i+j]] for j in range(seq_len)] for i in range(1,len(text)-seq_len)]

X_train = np.asarray(X_train).astype(np.int32)
Y_train = np.asarray(Y_train).astype(np.int32)

X_train = X_train[:(len(X_train)//b_size)*b_size]
Y_train = Y_train[:(len(Y_train)//b_size)*b_size]

X_train = np.reshape(X_train, [b_size, len(X_train)//b_size, seq_len])
Y_train = np.reshape(Y_train, [b_size, len(Y_train)//b_size, seq_len])

assert len(X_train) == len(Y_train)
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)


state_size = 256

idxs = tf.placeholder(tf.int32, [b_size, seq_len], name='words')
goals = tf.placeholder(tf.int32, [b_size, seq_len], name='goals')

embedding_matrix = tf.Variable(tf.random_normal((len(chars), emb_size), 0, 0.01), name='embedding_matrix')
emb_chars = tf.nn.embedding_lookup(embedding_matrix, idxs, name='emb_lookup')

bn_emb_chars = tf.layers.batch_normalization(emb_chars)

E = tf.Variable(tf.random_normal((emb_size,state_size), 0, 1./np.sqrt(emb_size)),
                                                                            name='E')
E_b = tf.Variable(tf.zeros([state_size]), name='E_b')


def lstm_vars(name):
    W = tf.Variable(tf.random_normal((4,state_size, state_size),0,1/np.sqrt(state_size)),
                                                                            name=name+'W')
    U = tf.Variable(tf.random_normal((4,state_size, state_size),0,1/np.sqrt(state_size)),
                                                                            name=name+'U')
    b = tf.Variable(tf.zeros([4,state_size]), name=name+'b')
    return W, U, b

lstm1_vars = lstm_vars('lstm1')
lstm2_vars = lstm_vars('lstm2')

V = tf.Variable(tf.random_normal((state_size, emb_size),0, 1/np.sqrt(state_size)),
                                                                    name='V')
V_b = tf.Variable(tf.zeros([emb_size]), name='V_b')

h_t = tf.Variable(tf.zeros([b_size, state_size], dtype=tf.float32),
                                                            name='h_t',
                                                            trainable=False)

c_t = tf.Variable(tf.zeros([b_size, state_size], dtype=tf.float32),
                                                            name='c_t',
                                                            trainable=False)

###############
# Graph

z1_dropout = tf.placeholder(tf.float32)
c1_dropout = tf.placeholder(tf.float32)
h1_dropout = tf.placeholder(tf.float32)

def fwd_prop(prevs, word):
    h_t_prev, c_t_prev = tf.split(prevs, 2)
    h_t_prev = tf.gather(h_t_prev, 0)
    c_t_prev = tf.gather(c_t_prev, 0)

    activs = tf.nn.relu(tf.matmul(word, E) + E_b)
    W, U, b = lstm1_vars

    f_t1 = tf.nn.sigmoid(tf.matmul(activs, W[0]) + tf.matmul(h_t_prev, U[0]) + b[0])
    i_t1 = tf.nn.sigmoid(tf.matmul(activs, W[1]) + tf.matmul(h_t_prev, U[1]) + b[1])
    o_t1 = tf.nn.sigmoid(tf.matmul(activs, W[2]) + tf.matmul(h_t_prev, U[2]) + b[2])
    z_t1 = tf.nn.sigmoid(tf.matmul(activs, W[3]) + tf.matmul(h_t_prev, U[3]) + b[3])

    z_t1 = tf.nn.dropout(z_t1,z1_dropout, name='z1dropout')

    c_t1 = f_t1*c_t_prev + i_t1 * z_t1
    h_t1 = o_t1 * tf.nn.tanh(c_t1)

    c_t1 = tf.nn.dropout(c_t1, c1_dropout, name='c1dropout')
    h_t1 = tf.nn.dropout(h_t1, h1_dropout, name='h1dropout')

    W, U, b = lstm2_vars

    f_t2 = tf.nn.sigmoid(tf.matmul(h_t1, W[0]) + tf.matmul(c_t1, U[0]) + b[0])
    i_t2 = tf.nn.sigmoid(tf.matmul(h_t1, W[1]) + tf.matmul(c_t1, U[1]) + b[1])
    o_t2 = tf.nn.sigmoid(tf.matmul(h_t1, W[2]) + tf.matmul(c_t1, U[2]) + b[2])
    z_t2 = tf.nn.sigmoid(tf.matmul(h_t1, W[3]) + tf.matmul(c_t1, U[3]) + b[3])

    c_t2 = f_t2 * c_t1 + i_t2*z_t2
    h_t2 = o_t2*tf.nn.tanh(c_t2)

    return [h_t2, c_t2]

transposed_emb_chars = tf.transpose(bn_emb_chars, [1,0,2], name='transpose1')
transposed_states = tf.scan(
    fwd_prop,
    transposed_emb_chars,
    initializer=[h_t, c_t]
)
h_states, c_states = tf.split(transposed_states, 2)
h_state_list = tf.split(h_states[0], seq_len)
c_state_list = tf.split(c_states[0], seq_len)

assign_h = h_t.assign(h_state_list[0][0])
assign_c = c_t.assign(c_state_list[0][0])
out_list = []
for s in h_state_list:
    out_list.append(tf.matmul(s[0],V)+V_b)

transposed_outputs = tf.stack(out_list, axis=0)
outputs = tf.transpose(transposed_outputs, [1,0,2], name='outputs')

logits = outputs
outputs = tf.nn.softmax(outputs)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=goals))
lr = tf.placeholder(tf.float32)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
zero_h = h_t.assign(tf.zeros([b_size, state_size],dtype=tf.float32))
zero_c = c_t.assign(tf.zeros([b_size, state_size],dtype=tf.float32))

saver = tf.train.Saver()

zs = np.zeros((b_size-1, seq_len))
def next_char(sequence, sess):
    seq = np.asarray([sequence[-seq_len:]])
    seq = np.concatenate([seq,zs])
    preds, ah, ac = sess.run([outputs, assign_h, assign_c], feed_dict={idxs:seq, z1_dropout:1.0, c1_dropout:1.0, h1_dropout:1.0})
    return np.random.choice(len(chars),1,p=preds[0,-1,:])[0]


#######
# Session
learn_rate = 0.001
save_file = './lstm_char.ckpt'

def zero_out(sess):
    sess.run(zero_h) # Resets state
    sess.run(zero_c)

with tf.Session() as sess:
    print("Begin Session")
    sess.run(tf.global_variables_initializer())
    zero_out(sess)
    epoch = 0
    while True:
        epoch += 1
        print("Epoch", epoch)
        running_avg = 0
        for i in range(len(X_train)//b_size):
            x = X_train[:,i,:]
            y = Y_train[:,i,:]
            cost, opt,ass_op_h,ass_op_c  = sess.run([loss, optimizer, assign_h, assign_c],
                                                 feed_dict={idxs:x, goals:y, lr:learn_rate,
                                                 z1_dropout:1., c1_dropout:.8, h1_dropout:1.})
            running_avg += cost
        print("Avg Cost:", running_avg/(len(X_train)//b_size), " "*20)
        if epoch % 70 == 0:
            zero_out(sess)
            basetime = time.time()
            print("Begin Sample:")
            seed = "He who fights with monsters might take care lest he thereby" +\
                        " become a monster. And if you gaze for long into an" +\
                        " abyss the abyss also gazes into you"
            sample = [char_idx[x] for x in list(seed)]
            for i in range(len(sample)-seq_len):
                next_char(sample[i:i+seq_len], sess)
            sample_len = 300
            for i in range(sample_len):
                sample.append(next_char(sample, sess))
            print(np.asarray(sample).shape)
            verbalization = [idx_char[i] for i in sample]
            print("Sample:\n",''.join(verbalization))
            print("Execution time:", time.time()-basetime)

        zero_out(sess)
        saver.save(sess, save_file)
