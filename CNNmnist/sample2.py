import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data  # 导入下载数据集手写体
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)


class CNNNet:  # 创建一个CNNNet类
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='input_x')  # 创建数据占位符
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='input_y')  # 创建标签占位符

        self.w1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 16], dtype=tf.float32, stddev=tf.sqrt(1 / 16), name='w1'))  # 定义 第一层/输入层/卷积层 w
        self.b1 = tf.Variable(tf.zeros(shape=[16], dtype=tf.float32, name='b1'))  # 定义 第一层/输入层/卷积层 偏值b

        self.w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], dtype=tf.float32, stddev=tf.sqrt(1 / 32), name='w2'))  # 定义 第二层/卷积层 w
        self.b2 = tf.Variable(tf.zeros(shape=[32], dtype=tf.float32, name='b2'))  # 定义 第二层/卷积层 偏值b

        self.fc_w1 = tf.Variable(tf.truncated_normal(shape=[28 * 28 * 32, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128), name='fc_w1'))  # 定义 第三层/全链接层/ w
        self.fc_b1 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32, name='fc_b1'))  # 定义 第三层/全链接层/ 偏值b

        self.fc_w2 = tf.Variable(tf.truncated_normal(shape=[128, 10], dtype=tf.float32, stddev=tf.sqrt(1 / 10), name='fc_w2'))  # 定义 第四层/全链接层/输出层 w
        self.fc_b2 = tf.Variable(tf.zeros(shape=[10], dtype=tf.float32, name='fc_b2'))  # 定义 第四层/全链接层/输出层 偏值b

	# 前向计算
    def forward(self):
        # 前向计算 第一层/输入层/卷积层
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x, self.w1, strides=[1, 1, 1, 1], padding='SAME', name='conv1') + self.b1)
        # 前向计算 第二层/卷积层
        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.conv1, self.w2, strides=[1, 1, 1, 1], padding='SAME', name='conv2') + self.b2)
        # 将第二层卷积后的数据撑开为 [批次, 数据]
        self.flat = tf.reshape(self.conv2, [-1, 28 * 28 * 32])
        # 前向计算 第三层/全链接层
        self.fc1 = tf.nn.relu(tf.matmul(self.flat, self.fc_w1) + self.fc_b1)
        # 前向计算 第四层/全链接层/输出层
        self.fc2 = tf.matmul(self.fc1, self.fc_w2) + self.fc_b2
        # 输出层 softmax分类
        self.output = tf.nn.softmax(self.fc2)

	# 后向计算
    def backward(self):
		# 定义 softmax交叉熵 求损失
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc2, labels=self.y))
        # 使用 AdamOptimizer优化器 优化cost 损失函数
        self.opt = tf.train.AdamOptimizer().minimize(self.cost)

	# 计算测试集识别精度
    def acc(self):
		# 将预测值 output 和 标签值 self.y 进行比较
        self.acc2 = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
		#  最后对比较出来的bool值 转换为float32类型后 求均值就可以看到满值为 1的精度显示
        self.accaracy = tf.reduce_mean(tf.cast(self.acc2, dtype=tf.float32))


if __name__ == '__main__':
    net = CNNNet()  # 启动tensorflow绘图的CNNNet
    net.forward()   # 启动前向计算
    net.backward()  # 启动后向计算
    net.acc()       # 启动精度计算
    init = tf.global_variables_initializer()  # 定义初始化tensorflow所有变量操作
    with tf.Session() as sess:  # 创建一个Session会话
        sess.run(init)          # 执行init变量内的初始化所有变量的操作
        for i in range(10000):  # 训练10000次
            ax, ay = mnist.train.next_batch(100)  # 从mnist数据集中取数据出来 ax接收图片 ay接收标签
            ax_batch = ax.reshape([-1, 28, 28, 1])  # 将取出的 图片数据 reshape成 NHWC 结构
            loss, output, accaracy, _ = sess.run(fetches=[net.cost, net.output, net.accaracy, net.opt], feed_dict={net.x: ax_batch, net.y: ay})  # 将数据喂进CNN网络
            # print(loss)      # 打印损失
            # print(accaracy)  # 打印训练精度
            if i % 10 == 0:    # 每训练10次
                test_ax, test_ay = mnist.test.next_batch(100)  # 则使用测试集对当前网络进行测试
                test_ax_batch = test_ax.reshape([-1, 28, 28, 1])  # 将取出的 图片数据 reshape成 NHWC 结构
                test_output = sess.run(net.output, feed_dict={net.x: test_ax_batch})  # 将测试数据喂进网络 接收一个output值
                test_acc = tf.equal(tf.argmax(test_output, 1), tf.argmax(test_ay, 1))  # 对output值和标签y值进行求比较运算
                accaracy2 = sess.run(tf.reduce_mean(tf.cast(test_acc, dtype=tf.float32)))  # 求出精度的准确率进行打印
                print(accaracy2)  # 打印当前测试集的精度
