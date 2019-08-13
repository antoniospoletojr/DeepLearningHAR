#!/usr/bin/env python
# coding: utf-8

# # HAR CNN training 

# In[1]:


# Imports
import numpy as np
import os
import argparse
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from utils.utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython import get_ipython
class CNN:
    def __init__(self, path_to_dataset):
        self.X_train, labels_train, list_ch_train = read_data(data_path=path_to_dataset, split="train")  # train
        self.X_test, labels_test, list_ch_test = read_data(data_path=path_to_dataset, split="test")      # test

        assert list_ch_train == list_ch_test, "Mistmatch in channels!"

        # Normalize
        self.X_train, self.X_test = standardize(self.X_train, self.X_test)

        # Train/Validation Split
        self.X_tr, self.X_vld, lab_tr, lab_vld = train_test_split(
            self.X_train, labels_train, stratify=labels_train, random_state=123)

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
        # Placeholders
        self.graph = tf.Graph()

        # Construct placeholders
        with self.graph.as_default():
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, self.n_channels], name = 'inputs')
            self.labels_ = tf.placeholder(tf.float32, [None, self.n_classes], name = 'labels')
            self.keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
            self.learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')


        # Build Convolutional Layers
        with self.graph.as_default():
            # (batch, 128, 9) --> (batch, 64, 18)
            conv1 = tf.layers.conv1d(inputs=self.inputs_, filters=18, kernel_size=2, strides=1,
                                     padding='same', activation = tf.nn.relu)
            max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

            # (batch, 64, 18) --> (batch, 32, 36)
            conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1,
                                     padding='same', activation = tf.nn.relu)
            max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

            # (batch, 32, 36) --> (batch, 16, 72)
            conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1,
                                     padding='same', activation = tf.nn.relu)
            max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

            # (batch, 16, 72) --> (batch, 8, 144)
            conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=2, strides=1,
                                     padding='same', activation = tf.nn.relu)
            max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

        # Now, flatten and pass to the classifier
        with self.graph.as_default():
            # Flatten and add dropout
            flat = tf.reshape(max_pool_4, (-1, 8*144))
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

        self.session = tf.Session(graph=self.graph)
        self.freeze_the_graph("har")


    def train_network(self):
        # Train the network
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
                for x,y in get_batches(self.X_tr, self.y_tr, self.batch_size):
                    # Feed dictionary
                    feed = {self.inputs_ : x, self.labels_ : y, self.keep_prob_ : 0.5, self.learning_rate_ : self.learning_rate}
                    # Loss
                    loss, _ , acc = sess.run([self.cost, self.optimizer, self.accuracy], feed_dict = feed)
                    train_acc.append(acc)
                    train_loss.append(loss)
                    # Print at each 5 iters
                    if (iteration % 5 == 0):
                        print("Epoch: {}/{}".format(e, self.epochs),
                              "Iteration: {:d}".format(iteration),
                              "Train loss: {:6f}".format(loss),
                              "Train acc: {:.6f}".format(acc))

                    # Compute validation loss at every 10 iterations
                    if (iteration%10 == 0):
                        val_acc_ = []
                        val_loss_ = []

                        for x_v, y_v in get_batches(self.X_vld, self.y_vld, self.batch_size):
                            # Feed
                            feed = {self.inputs_ : x_v, self.labels_ : y_v, self.keep_prob_ : 1.0}
                            # Loss
                            loss_v, acc_v = sess.run([self.cost, self.accuracy], feed_dict = feed)
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
            self.saver.save(sess,"checkpoints-cnn/har.ckpt")

        # Plot training and test loss
        t = np.arange(iteration-1)
        plt.figure(figsize = (6,6))
        plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
        plt.xlabel("iteration")
        plt.ylabel("Loss")
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        # Plot Accuracies
        plt.figure(figsize = (6,6))
        plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
        plt.xlabel("iteration")
        plt.ylabel("Accuray")
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()


    def freeze_the_graph(self, model_name):
        from tensorflow.python.tools import freeze_graph
        save_path = "./checkpoints-cnn/"  # directory to model files
        tf.train.write_graph(self.session.graph_def, save_path, "savegraph.pbtxt")
        # Freeze the graph
        input_graph_path = save_path + 'savegraph.pbtxt'  # complete path to the input graph
        checkpoint_path = save_path + 'har.ckpt'  # complete path to the model's checkpoint file
        input_saver_def_path = ""
        input_binary = False
        output_node_names = "labels"  # output node's name. Should match to that mentioned in your code
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_frozen_graph_name = save_path + 'frozen_model_' + model_name + '.pb'  # the name of .pb file you would like to give
        clear_devices = True
        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                  input_binary, checkpoint_path, output_node_names,
                                  restore_op_name, filename_tensor_name,
                                  output_frozen_graph_name, clear_devices, "")



    # Evaluate on test set
    def evaluate_on_test_set(self):
        test_acc = []
        # Create some variables.
        with self.session as sess:
            # Restore
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))
            #print(self.graph.get_tensor_by_name('labels:0'))
            #print(self.graph.get_tensor_by_name('inputs:0'))
            #export_dir = "saved_model"
            #tf.lite.TFLiteConverter.from_session(sess, self.inputs_, self.logits)
            for x_t, y_t in get_batches(self.X_test, self.y_test, self.batch_size):
                feed = {self.inputs_: x_t,
                        self.labels_: y_t,
                        self.keep_prob_: 1}

                batch_acc = sess.run(self.accuracy, feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy: {:.6f}".format(np.mean(test_acc)))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    print("--- Initializing CNN ---")
    cnn = CNN("./UCIHAR/")
    print("--- Building Graph ---")
    cnn.build_graph()
    #print("--- Training Network ---")
    #cnn.train_network()

    print("--- Evaluating on Test-Set ---")
    cnn.evaluate_on_test_set()

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--model_dir", type=str, default="lol", help="Model folder to export")
    #parser.add_argument("--output_node_names", type=str, default="asd",
    #                    help="The name of the output nodes, comma separated.")
    #args = parser.parse_args()
    #print(args.output_node_names)
    #freeze_graph(args.model_dir, args.output_node_names)
    #freeze_graph("checkpoints-cnn")

def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def
