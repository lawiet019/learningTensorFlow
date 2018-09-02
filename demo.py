import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# the first session
# v1 = tf.constant([[2,3]])
# v2 = tf.constant([[2],[3]])
# product = tf.matmul(v1,v2)
# print(product)
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

#the session add 10 to num every time
# num = tf.Variable(0,name="count")
# new_value = tf.add(num,10)
# op = tf.assign(num,new_value)
# with tf.Session() as sess:
#     #初始化全局变量
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(num))
#     for i in range(5):
#         sess.run(op)
#         print(sess.run(num))


# create variable placeholder
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# new_value = tf.multiply(input1,input2)
# with tf.Session() as sess:
#     print(sess.run(new_value,feed_dict={input1:1,input2:2}))

#https://blog.csdn.net/simplehouse/article/details/78348298


import tensorflow as tf
x = tf.placeholder(tf.float32,shape=(1,2))
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a = tf.matmul(x,w1)
b = tf.matmul(a,w2)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    result = sess.run(b,feed_dict={x:[[0.5,0.7]]})
print(result)

import numpy as np
Batch_Size = 5
seed = 23455
rng = np.random.RandomState(seed)
X= rng.rand(32,2)
Y = [[(int(x0+x1<1))] for (x0,x1) in X]

x = tf.placeholder(tf.float32,shape=(None,2))
y = tf.placeholder(tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a = tf.matmul(x,w1)
b = tf.matmul(a,w2)

loss = tf.reduce_mean(tf.square(b-y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    STEPS = 3000
    for i in range(STEPS):
        start = (i*Batch_Size) %32
        end = start + Batch_Size
        sess.run(train_step,feed_dict={x:X[start:end],y:Y[start:end]})
        if i%500 ==0:
            total_loss=sess.run(loss,feed_dict={x:X,y:Y})
            print(i,total_loss)
    print(sess.run(w1))
    print(sess.run(w2))
