import tensorflow as tf
import numpy as np

x = tf.placeholder(np.float32, [None, 2])
W = tf.Variable(tf.zeros([2,1]))
b = tf.Variable(tf.zeros([1]))

y = tf.nn.sigmoid(tf.matmul(x, W) + b)

y_ = tf.placeholder(np.float32, [None, 1])

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
mean_square = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(mean_square)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [1]])

for i in xrange(10000):
	sess.run(train_step, feed_dict = {x: x_data, y_: y_data})

print sess.run(y, feed_dict = {x: x_data})
