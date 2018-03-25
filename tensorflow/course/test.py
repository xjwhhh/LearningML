import tensorflow as tf

xs = tf.placeholder(tf.float32, [None, 1], name="xs")
ys = tf.placeholder(tf.float32, [None, 1], name="ys")
Weights1 = tf.Variable(tf.constant([[0.0004]]), dtype=tf.float32, name="Weights1")
Biases1 = tf.Variable(tf.zeros([1, 1]) + 0.1, dtype=tf.float32, name="Biases1")
Wx_plus_b1 = tf.add(tf.matmul(xs, Weights1, name="matmul"), Biases1, name="add")
l1 = tf.nn.sigmoid(Wx_plus_b1)

Weights2 = tf.Variable(tf.constant([[10000.0]]), dtype=tf.float32, name="Weights2")
Biases2 = tf.Variable(tf.constant(-4999.5), dtype=tf.float32, name="Biases2")
Wx_plus_b2 = tf.add(tf.matmul(l1, Weights2, name="matmul"), Biases2, name="add")

prediction = Wx_plus_b2

loss = tf.reduce_mean(tf.square(tf.subtract(ys, prediction, name="Sub"), name="Square"), name="ReduceMean")

# tf.train.GradientDescentOptimizer,实现梯度下降算法的优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss,
                                                             name="minimize")
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for i in range(1000):
        a, b, c, d, e, f = sess.run([train_step, loss, Weights1, Biases1, Weights2, Biases2],
                                    feed_dict={xs: [[i], [i]], ys: [[i], [i]]})
        print(b)
        print(c)
        print(d)
        print(e)
        print(f)
        print("22222")
    writer.close()
