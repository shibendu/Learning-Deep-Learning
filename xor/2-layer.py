import tensorflow as tf
import numpy as np

x = tf.placeholder(np.float32, [None, 2])
y_ = tf.placeholder(np.float32, [None, 1])

W_h1 = tf.Variable(tf.random_uniform([2, 20], -.01, .01))
b_h1 = tf.Variable(tf.random_uniform([20], -.01, .01))

#W_h2 = tf.Variable(tf.random_normal([2, 2], stddev=0.01))
#b_h2 = tf.Variable(tf.random_normal([2], stddev=0.01))

W_o = tf.Variable(tf.random_uniform([20, 1], -.01, .01))
b_o = tf.Variable(tf.random_uniform([1], -.01, .01))

y_h1 = tf.nn.relu(tf.matmul(x, W_h1) + b_h1)
#y_h2 = tf.nn.softmax(tf.matmul(y_h1, W_h2) + b_h2)
y = tf.nn.sigmoid(tf.matmul(y_h1, W_o) + b_o)

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
mean_square = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(mean_square)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

#print sess.run(W_h1)
#print sess.run(b_h1)

#print sess.run(W_o)
#print sess.run(b_o)

for step in xrange(100000):
	#feed_dict = {x: x_data, y_:y_data}
    e, a = sess.run([mean_square,train_step], feed_dict = {x: x_data, y_: y_data})
    if e<0.001:
    	print "yay"
    	break
    print "step %d : entropy %s" % (step,e)
	#sess.run(train_step, feed_dict = {x: x_data, y_: y_data})

print sess.run(y, feed_dict = {x: x_data})