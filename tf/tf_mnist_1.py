from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#加载预定义好格式的样本数据（可参考）

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #one_hot参数代表的是类别是一个one_hot数组
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

#真实概率分布y_
y_ = tf.placeholder(tf.float32, [None, 10])
# 损失函数的定义
# reduce sum 求和
# reduce mean 对每个batch数据求平均值
# reduction_indices 按照行还是列
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

# 优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 参数初始化
tf.global_variables_initializer().run()
# 使用一小部分的数据进行随机梯度下降
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_:batch_ys})

# 效果评价
# 看概率最大的那个数值是否与实际的值相等
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 用测试集的数据进行验证
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))
