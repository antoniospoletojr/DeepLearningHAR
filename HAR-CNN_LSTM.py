#!/usr/bin/env python
# coding: utf-8

# # HAR CNN + LSTM training 

# In[1]:


# Imports
import numpy as np
import os
from utils.utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import get_ipython


class CNN_LSTM:
    def __init__(self, path_to_dataset):
        self.X_train, labels_train, list_ch_train = read_data(data_path=path_to_dataset, split="train")  # train
        self.X_test, labels_test, list_ch_test = read_data(data_path=path_to_dataset, split="test")  # test

        assert list_ch_train == list_ch_test, "Mistmatch in channels!"

        # Standardize
        self.X_train, self.X_test = standardize(self.X_train, self.X_test)

        # Train/Validation Split
        self.X_tr, self.X_vld, lab_tr, lab_vld = train_test_split(self.X_train, labels_train,
                                                        stratify=labels_train, random_state=123)
        # One-hot encoding:
        self.y_tr = one_hot(lab_tr)
        self.y_vld = one_hot(lab_vld)
        self.y_test = one_hot(labels_test)

        # Hyperparameters
        self.lstm_size = 27  # 3 times the amount of channels
        self.lstm_layers = 2  # Number of layers
        self.batch_size = 600  # Batch size
        self.seq_len = 128  # Number of steps
        self.learning_rate = 0.0001  # Learning rate (default is 0.001)
        self.epochs = 1000

        # Fixed
        self.n_classes = 6
        self.n_channels = 9

    # Construct the graph
    def build_graph(self):
        self.graph = tf.Graph()

        # Construct placeholders
        with self.graph.as_default():
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, self.n_channels], name='inputs')
            self.labels_ = tf.placeholder(tf.float32, [None, self.n_classes], name='labels')
            self.keep_prob_ = tf.placeholder(tf.float32, name='keep')
            self.learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

        # Build Convolutional Layer(s)
        #
        # Questions:
        # * Should we use a different activation? Like tf.nn.tanh?
        # * Should we use pooling? average or max?

        # Convolutional layers
        with self.graph.as_default():
            # (batch, 128, 9) --> (batch, 128, 18)
            conv1 = tf.layers.conv1d(inputs=self.inputs_, filters=18, kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            n_ch = self.n_channels * 2

        # Now, pass to LSTM cells
        with self.graph.as_default():
            # Construct the LSTM inputs and LSTM cells
            lstm_in = tf.transpose(conv1, [1, 0, 2])  # reshape into (seq_len, batch, channels)
            lstm_in = tf.reshape(lstm_in, [-1, n_ch])  # Now (seq_len*N, n_channels)
            # To cells
            lstm_in = tf.layers.dense(lstm_in, self.lstm_size, activation=None)  # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?
            # Open up the tensor into a list of seq_len pieces
            lstm_in = tf.split(lstm_in, self.seq_len, 0)
            # Add LSTM layers
            lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob_)
            self.cell = tf.contrib.rnn.MultiRNNCell([drop] * self.lstm_layers)
            self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        # Define forward pass and cost function:
        with self.graph.as_default():
            outputs, self.final_state = tf.contrib.rnn.static_rnn(self.cell, lstm_in, dtype=tf.float32,
                                                             initial_state= self.initial_state)
            # We only need the last output tensor to pass into a classifier
            self.logits = tf.layers.dense(outputs[-1], self.n_classes, name='logits')

            # Cost function and optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_))
            # optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost) # No grad clipping

            # Grad clipping
            train_op = tf.train.AdamOptimizer(self.learning_rate_)
            gradients = train_op.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
            self.optimizer = train_op.apply_gradients(capped_gradients)

            # Accuracy
            correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        with self.graph.as_default():
            self.saver = tf.train.Saver()

    def train_network(self):
        if (os.path.exists('checkpoints-crnn') == False):
            get_ipython().system('mkdir checkpoints-crnn')

        validation_acc = []
        validation_loss = []
        train_acc = []
        train_loss = []

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1

            for e in range(self.epochs):
                # Initialize
                state = sess.run(self.initial_state)

                # Loop over batches
                for x, y in get_batches(self.X_tr, self.y_tr, self.batch_size):

                    # Feed dictionary
                    feed = {self.inputs_: x, self.labels_: y, self.keep_prob_: 0.5,
                            self.initial_state: state, self.learning_rate_: self.learning_rate}

                    loss, _, state, acc = sess.run([self.cost, self.optimizer, self.final_state, self.accuracy],
                                                   feed_dict=feed)
                    train_acc.append(acc)
                    train_loss.append(loss)

                    # Print at each 5 iters
                    if (iteration % 5 == 0):
                        print("Epoch: {}/{}".format(e, self.epochs),
                              "Iteration: {:d}".format(iteration),
                              "Train loss: {:6f}".format(loss),
                              "Train acc: {:.6f}".format(acc))

                    # Compute validation loss at every 25 iterations
                    if (iteration % 25 == 0):

                        # Initiate for validation set
                        val_state = sess.run(self.cell.zero_state(self.batch_size, tf.float32))

                        val_acc_ = []
                        val_loss_ = []
                        for x_v, y_v in get_batches(self.X_vld, self.y_vld, self.batch_size):
                            # Feed
                            feed = {self.inputs_: x_v, self.labels_: y_v, self.keep_prob_: 1.0, self.initial_state: val_state}

                            # Loss
                            loss_v, state_v, acc_v = sess.run([self.cost, self.final_state, self.accuracy], feed_dict=feed)

                            val_acc_.append(acc_v)
                            val_loss_.append(loss_v)

                        # Print info
                        print("Epoch: {}/{}".format(e, self.epochs),
                              "Iteration: {:d}".format(iteration),
                              "Validation loss: {:6f}".format(np.mean(val_loss_)),
                              "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                        # Store
                        validation_acc.append(np.mean(val_acc_))
                        validation_loss.append(np.mean(val_loss_))

                    # Iterate
                    iteration += 1

            self.saver.save(sess, "checkpoints-crnn/har.ckpt")

        # Plot training and test loss
        t = np.arange(iteration - 1)
        plt.figure(figsize=(6, 6))
        plt.plot(t, np.array(train_loss), 'r-', t[t % 25 == 0], np.array(validation_loss), 'b*')
        plt.xlabel("iteration")
        plt.ylabel("Loss")
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

        # Plot Accuracies
        plt.figure(figsize=(6, 6))
        plt.plot(t, np.array(train_acc), 'r-', t[t % 25 == 0], validation_acc, 'b*')
        plt.xlabel("iteration")
        plt.ylabel("Accuray")
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

    def recover_graph(self):
        with tf.Session() as sess:
            self.saver = tf.train.import_meta_graph('checkpoints-crnn/har.ckpt.meta')
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints-crnn'))
            print(sess.run('labels:0'))


    # Evaluate on test set
    def evaluate_on_test_set(self):
        test_acc = []
        with tf.Session(graph=self.graph) as sess:
            # Restore
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints-crnn'))

            for x_t, y_t in get_batches(self.X_test, self.y_test, self.batch_size):
                feed = {self.inputs_: x_t,
                        self.labels_: y_t,
                        self.keep_prob_: 1}

                batch_acc = sess.run(self.accuracy, feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy: {:.6f}".format(np.mean(test_acc)))



if __name__ == "__main__":
    print("--- Initializing CNN-LSTM ---")
    cnn_lstm = CNN_LSTM("./UCIHAR/")


    print("--- Building Graph ---")
    cnn_lstm.build_graph()

    #print("--- Training Network ---")
    #cnn_lstm.train_network()

    print("--- Evaluating on Test-Set ---")
    cnn_lstm.evaluate_on_test_set()

