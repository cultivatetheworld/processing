""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import csv as csv
import datetime
import numpy as np

# Import dataset
# from google.colab import files
# uploaded = files.upload()

import io
df2 = csv.reader(open("rdu-weather-history.csv", 'r'))
head = next(df2)
use = True
n = 0
input_ml = []
output_ml = []
for row in df2:
  if(n>2):
    break
  use = True
  # print(row[0])
  a = row[0].split(";")
  for i in range(0, len(a)):
    if (a[i]=="Yes"):
      a[i]=1
    elif (a[i]=="No"):
      a[i]=0
    if(i>0):
      try:
        a[i]=float(a[i])
      except(ValueError):
        use = False
  t = a[0].split("-")
  tboi = datetime.datetime(int(t[0]),int(t[1]), int(t[2]))
  if(n==0):
    start_time = tboi
  dboi = (tboi-start_time).total_seconds()/(60*60*24)
  temp = []
  temp.append(dboi)
  if(use):
    input_ml.append(np.array(temp+a[1:3]+a[4:]))
    output_ml.append(a[3])
  n= n+1
input_ml = np.array(input_ml)
output_ml = np.array(output_ml)

tf.reset_default_graph()

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 27
display_step = 200

# Network Parameters
num_input = 27 # data input 27 values
timesteps = 27
num_hidden = 128 # hidden layer num of features
num_classes =  1

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(0, training_steps):
        batch_x, batch_y = input_ml[step:step+batch_size], output_ml[step:batch_size]
        # print(batch_x)
        batch_x = batch_x.reshape((1, timesteps, num_input.reshape((-1,1))))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 bois
    test_len = 128
    test_data = input_ml[0:128]
    test_label = output_ml[0:128]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
