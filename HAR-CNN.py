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
            tf.identity(self.logits, name="logits")

            # Cost function and optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate_).minimize(self.cost)

            # Accuracy
            correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        with self.graph.as_default():
            self.saver = tf.train.Saver()

        self.session = tf.Session(graph=self.graph)


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
        output_node_names = "logits"    # output node's name. Should match to that mentioned in your code
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_frozen_graph_name = save_path + 'frozen_model_' + model_name + '.pb'  # the name of .pb file you would like to give
        clear_devices = True

        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                  input_binary, checkpoint_path, output_node_names,
                                  restore_op_name, filename_tensor_name,
                                  output_frozen_graph_name, clear_devices, "")

        graph_def_file = output_frozen_graph_name

        input_arrays = ["inputs"]
        output_arrays = ["logits"]
        converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
        #tflite_model = converter.convert()  # qui nascono i problemi!

        ''' NON WORKA: Some of the operators in the model are not supported by the standard TensorFlow Lite runtime. 
        output_arrays = ["logits"]
        input_arrays = {"inputs", "keep"}
        converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays, input_shapes={"keep":[1]})
        tflite_model = converter.convert()     #qui nascono i problemi!
        '''
        #open("converted_model.tflite", "wb").write(tflite_model)

        #tf.saved_model.simple_save(self.session, "./lol/", inputs={'x':self.inputs_}, outputs={"y":self.logits})

        # Converting a GraphDef from session.
        #converter = tf.lite.TFLiteConverter.from_session(self.session, self.inputs_, self.logits)
        #tflite_model = converter.convert()
        #open("converted_model.tflite", "wb").write(tflite_model)

        #tf.lite.toco_convert(self.session.graph_def, [self.inputs_], [self.logits])

        #writer = tf.summary.FileWriter('./named_scope', self.session.graph)
        #writer.close()

    # Evaluate on test set
    def evaluate_on_test_set(self):
        test_acc = []
        # Create some variables.
        with self.session as sess:
            # Restore
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))

            # Run through batches
            for x_t, y_t in get_batches(self.X_test, self.y_test, self.batch_size):
                feed = {self.inputs_: x_t,
                        self.labels_: y_t,
                        self.keep_prob_: 1}
                batch_acc = sess.run(self.accuracy, feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy: {:.6f}".format(np.mean(test_acc)))

    def re_create_the_network_and_test(self):
        test_acc = []
        # Create some variables.
        with tf.Session() as sess:
            # Restore the network
            new_saver = tf.train.import_meta_graph('./checkpoints-cnn/har.ckpt.meta')

            # Load the parameters
            new_saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))
            graph = tf.get_default_graph()

            # Access the tensors
            inputs = graph.get_tensor_by_name('inputs:0')
            labels = graph.get_tensor_by_name('labels:0')
            keep = graph.get_tensor_by_name('keep:0')
            accuracy = graph.get_tensor_by_name('accuracy:0')

            for x_t, y_t in get_batches(self.X_test, self.y_test, self.batch_size):
                feed = {inputs: x_t,
                        labels: y_t,
                        keep: 1}
                batch_acc = sess.run(accuracy, feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy2: {:.6f}".format(np.mean(test_acc)))

    def re_create_network_from_frozen_graph(self, frozen_graph_filename):
        print("--- Accessing {} --- ".format(frozen_graph_filename))
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
            graph = tf.get_default_graph()

            # Salvataggio del grafo
            #with tf.Session() as sess:
                #writer = tf.summary.FileWriter('./named_scope', sess.graph)
                #writer.close()

            graph = tf.get_default_graph()

            # Access the tensors
            inputs = graph.get_tensor_by_name('import/inputs:0')
            keep = graph.get_tensor_by_name('import/keep:0')
            logits = graph.get_tensor_by_name('import/logits:0')

            # Gather the activities
            activities = []
            with open("./UCIHAR/activity_labels.txt") as file:
                for line in file:
                    field = line.split(' ')
                    field[1] = field[1][:-1]
                    activities.append(field[1])
            #print("Labels: {}\n".format(activities))

            with tf.Session() as sess:
                i=0
                for x_t, _ in get_batches(self.X_test, [], 1):  #PRIMA: for x_t, y_t in get_batches(self.X_test, self.y_test, 1):
                    feed = {inputs: x_t,
                            keep: 1}
                    classification_tensor = logits  # teoricamente -> tf.nn.softmax(logits)
                    prediction = sess.run(classification_tensor, feed_dict=feed)
                    index_p = np.argmax(prediction)
                    #print("iter-{} Predicted {} -> {} | real {} -> {}".format(i,index_p,activities[index_p], np.argmax(y_t), activities[np.argmax(y_t)]))
                    i=i+1

if __name__ == "__main__":
    #tf.logging.set_verbosity(tf.logging.ERROR)
    print("--- Initializing CNN ---")
    cnn = CNN("./UCIHAR/")

    print("--- Building the Graph ---")
    cnn.build_graph()

    # print("--- Training Network ---")
    # cnn.train_network()

    print("--- Freezing the Graph ---")
    cnn.freeze_the_graph("har")

    print("--- Evaluating on Test-Set ---")
    cnn.evaluate_on_test_set()

    print("\n--- Re-create the network from Frozen-Graph and test it on Test-Set ---")
    cnn.re_create_network_from_frozen_graph("./checkpoints-cnn/frozen_model_har.pb")    #location del modello freezato

    print("--- DONE ---")
    ''' WORKING (Ricreare la rete a partire dal .meta e il .data)
    print("\n--- Re-create the network and test it on Test-Set ---")
    #You find a tutorial here: https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    cnn.re_create_the_network_and_test()
    '''
