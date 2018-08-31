import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
new_value = tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run(new_value,feed_dict={input1:1,input2:2}))

#https://blog.csdn.net/simplehouse/article/details/78348298
