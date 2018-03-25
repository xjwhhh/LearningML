from __future__ import print_function, division
import numpy as np
import tensorflow as tf

v = tf.Variable(initial_value=[1, 2])
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 显式地传递session到函数里面
    v.load(value=[3, 4], session=sess)
    print(v.eval(session=sess))
