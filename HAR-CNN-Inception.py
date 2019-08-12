#!/usr/bin/env python
# coding: utf-8
# HAR CNN training 

# Imports
import numpy as np
import os
from utils.utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython import get_ipython
import tensorflow as tf

class CNN_Inception:
    def __init__(self, path_to_dataset):
        self.X_train, labels_train, list_ch_train = read_data(data_path=path_to_dataset, split="train")  # train
        self.X_test, labels_test, list_ch_test = read_data(data_path=path_to_dataset, split="test")  # test

        assert list_ch_train == list_ch_test, "Mistmatch in channels!"
        # Normalize
        self.X_train, self.X_test = standardize(self.X_train, self.X_test)

        # Train/Validation Split
        self.X_tr, self.X_vld, lab_tr, lab_vld = train_test_split(self.X_train, labels_train,
                                                        stratify=labels_train, random_state=123)
        # One-hot encoding:
        self.y_tr = one_hot(lab_tr)
        self.y_vld = one_hot(lab_vld)
        self.y_test = one_hot(labels_test)

        # Hyperparameters
        self.batch_size = 600  # Batch size
        self.seq_len = 128  # Number of steps
        self.learning_rate = 0.0001
        self.epochs = 1000
        self.n_classes = 6
        self.n_channels = 9

    #Construct the graph
    def build_graph(self):
        self.graph = tf.Graph()

        # Construct placeholders
        with self.graph.as_default():
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, self.n_channels], name='inputs')
            self.labels_ = tf.placeholder(tf.float32, [None, self.n_classes], name='labels')
            self.keep_prob_ = tf.placeholder(tf.float32, name='keep')
            self.learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

        # Build Convolutional Layers
        with self.graph.as_default():
            # (batch, 128, 9) --> (batch, 64, 18)
            conv1 = tf.layers.conv1d(inputs=self.inputs_, filters=18, kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

            # (batch, 64, 18) --> (batch, 32, 18)
            conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=18, kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

            # (batch, 32, 18) --> (batch, 16, 36)
            conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=36, kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

            # (batch, 16, 36) --> (batch, 8, 36)
            conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=36, kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

        # Build the inception layer:
        with self.graph.as_default():
            # convolution: input to output of inception (size=1)
            # (batch, 8, 36) --> (batch, 8, 36)
            conv1_11 = tf.layers.conv1d(inputs=max_pool_4, filters=36, kernel_size=1, strides=1,
                                        padding='same', activation=tf.nn.relu)

            # convolution: input to middle layer of inception (size=1)
            # (batch, 8, 36) --> (batch, 8, 18)
            conv1_21 = tf.layers.conv1d(inputs=max_pool_4, filters=18, kernel_size=1, strides=1,
                                        padding='same', activation=tf.nn.relu)

            # convolution: input to middle layer of inception (size=1)
            # (batch, 8, 36) --> (batch, 8, 18)
            conv1_31 = tf.layers.conv1d(inputs=max_pool_4, filters=18, kernel_size=1, strides=1,
                                        padding='same', activation=tf.nn.relu)

            # average pool: input to middle layer of inception
            # (batch, 8, 36) --> (batch, 8, 36)
            avg_pool_41 = tf.layers.average_pooling1d(inputs=max_pool_4, pool_size=2, strides=1, padding='same')

            ## Middle layer of inception

            # convolution: middle to out layer of inception (size=2)
            # (batch, 8, 18) --> (batch, 8, 36)
            conv2_22 = tf.layers.conv1d(inputs=conv1_21, filters=36, kernel_size=2, strides=1,
                                        padding='same', activation=tf.nn.relu)

            # convolution: middle to out layer of inception (size=4)
            # (batch, 8, 18) --> (batch, 8, 36)
            conv4_32 = tf.layers.conv1d(inputs=conv1_31, filters=36, kernel_size=4, strides=1,
                                        padding='same', activation=tf.nn.relu)

            # convolution: middle to out layer of inception (size=1)
            # (batch, 8, 36) --> (batch, 8, 36)
            conv1_42 = tf.layers.conv1d(inputs=avg_pool_41, filters=36, kernel_size=1, strides=1,
                                        padding='same', activation=tf.nn.relu)

            ## Out layer: Concatenate filters
            # (batch, 8, 4*36)
            inception_out = tf.concat([conv1_11, conv2_22, conv4_32, conv1_42], axis=2)

        # Now, flatten and pass to the classifier
        with self.graph.as_default():
            # Flatten and add dropout
            flat = tf.reshape(inception_out, (-1, 8 * 144))
            flat = tf.nn.dropout(flat, keep_prob=self.keep_prob_)

            # Predictions
            self.logits = tf.layers.dense(flat, self.n_classes)

            # Cost function and optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate_).minimize(self.cost)

            # Accuracy
            correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        with self.graph.as_default():
            self.saver = tf.train.Saver()

    def train_network(self):
        if (os.path.exists('checkpoints-cnn') == False):
            get_ipython().system('mkdir checkpoints-cnn')

        validation_acc = []
        validation_loss = []

        train_acc = []
        train_loss = []

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1

            # Loop over epochs
            for e in range(self.epochs):

                # Loop over batches
                for x, y in get_batches(self.X_tr, self.y_tr, self.batch_size):

                    # Feed dictionary
                    feed = {self.inputs_: x, self.labels_: y, self.keep_prob_: 0.5, self.learning_rate_: self.learning_rate}

                    # Loss
                    loss, _, acc = sess.run([self.cost, self.optimizer, self.accuracy], feed_dict=feed)
                    train_acc.append(acc)
                    train_loss.append(loss)

                    # Print at each 5 iters
                    if (iteration % 5 == 0):
                        print("Epoch: {}/{}".format(e, self.epochs),
                              "Iteration: {:d}".format(iteration),
                              "Train loss: {:6f}".format(loss),
                              "Train acc: {:.6f}".format(acc))

                    # Compute validation loss at every 10 iterations
                    if (iteration % 10 == 0):
                        val_acc_ = []
                        val_loss_ = []

                        for x_v, y_v in get_batches(self.X_vld, self.y_vld, self.batch_size):
                            # Feed
                            feed = {self.inputs_: x_v, self.labels_: y_v, self.keep_prob_: 1.0}

                            # Loss
                            loss_v, acc_v = sess.run([self.cost, self.accuracy], feed_dict=feed)
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

            self.saver.save(sess, "checkpoints-cnn/har.ckpt")

        # Plot training and test loss
        t = np.arange(iteration - 1)

        plt.figure(figsize=(6, 6))
        plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
        plt.xlabel("iteration")
        plt.ylabel("Loss")
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

        # Plot Accuracies
        plt.figure(figsize=(6, 6))
        plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
        plt.xlabel("iteration")
        plt.ylabel("Accuray")
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()


    # Evaluate on test set
    def evaluate_on_test_set(self):
        test_acc = []
        with tf.Session(graph=self.graph) as sess:
            # Restore
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))

            for x_t, y_t in get_batches(self.X_test, self.y_test, self.batch_size):
                feed = {self.inputs_: x_t,
                        self.labels_: y_t,
                        self.keep_prob_: 1}

                batch_acc = sess.run(self.accuracy, feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy: {:.6f}".format(np.mean(test_acc)))

        print(self.y_test)
        # print(labels_test)
        activities = []
        with open("./UCIHAR/activity_labels.txt") as file:
            for line in file:
                field = line.split(' ')
                field[1] = field[1][:-1]
                activities.append(field[1])
        print("Labels: {}\n".format(activities))
        with tf.Session(graph=self.graph) as sess:
            # Restore
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))
            i = wrong_pred = 0
            test_set_len = len(self.X_test)
            print("------- Scanning Test Set (#elements: {}) one-by-one --------".format(test_set_len))
            # print("- Displaying wrong predictions ONLY: ")
            for x_t, y_t in get_batches(self.X_test, self.y_test, 1):
                feed = {self.inputs_: x_t,
                        self.labels_: y_t,
                        self.keep_prob_: 1}
                classification_tensor = self.logits  # teoricamente -> tf.nn.softmax(logits)
                prediction = sess.run(classification_tensor, feed_dict=feed)
                index_p = np.argmax(prediction)
                # print("{} <- {}\t|\tpredicted {}\n\t{}\t{}\n".format(activities[np.argmax(y_t)],np.argmax(y_t), index_p, y_t, np.round(prediction)))
                # print("iter-{} Predicted {} -> {} | real {} -> {}".format(i,index_p,activities[index_p], np.argmax(y_t), activities[np.argmax(y_t)]))
                if index_p != np.argmax(y_t):
                    # print("iter-{} Predicted {} -> {} | real {} -> {}".format(i,index_p,activities[index_p], np.argmax(y_t), activities[np.argmax(y_t)]))
                    wrong_pred = wrong_pred + 1
                i = i + 1
            print("\n------------------")
            print("Encountered 'wrong predictions' {} over {} total".format(wrong_pred, test_set_len))
            print("Home-made accuracy: {:.2f}%".format((test_set_len - wrong_pred) * 100 / test_set_len))
            print("------------------")


if __name__ == "__main__":
    print("--- Initializing CNN-Inception ---")
    cnn_inception = CNN_Inception("./UCIHAR/")

    print("--- Building Graph ---")
    cnn_inception.build_graph()

    #print("--- Training Network ---")
    #cnn_inception.train_network()

    print("--- Evaluating on Test-Set ---")
    cnn_inception.evaluate_on_test_set()





