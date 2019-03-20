# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:22:48 2019

@author: len
"""
# 计算图
# 计算图只包含计算步骤，不包含结果
import tensorflow as tf
'''two_node = tf.constant(2)
another_two_node = tf.constant(2)
two_node = tf.constant(2)
tf.constant(3)
print(two_node)'''
# 会话，会话的作用是处理内存分配和优化，是我们能够实际计算执行由图形指定的计算
'''two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
sess = tf.Session()
print(sess.run([two_node,sum_node]))'''
# sess.run() 调用它的次数越少越好，可以的话在一个 sess.run() 调用中返回多个项目，而不是进行多个调用。
# 占位符，一种用于接受外部输入的节点
'''input_placeholder = tf.placeholder(tf.int32)
sess = tf.Session()
print(sess.run(input_placeholder,feed_dict={input_placeholder: 2}))'''
# 使用sess.run()的deed_dict属性，来提供一个值
# 计算路径，仅通过必需的节点自动路由计算
# 创建变量 tf.get_variable() 前两个参数是必须的，其余是可选的
# tf.get_variable(name,shape) name唯一标识这个变量对象的字符串,shape一个与张量形状相对应的整数数组
'''count_variable = tf.get_variable("count",[])
sess = tf.Session()
print(sess.run(count_variable))'''
# 一个变量节点在首次创建时，它的值基本上就是null，任何尝试对它进行计算的操作都会抛出这个一场。
# 只能先给一个变量赋值后才能用它做计算
# 给变量赋值的两种方法：初始化器和 tf.assign()
'''count_variable = tf.get_variable("count1",[])#[]表示创建一个标量 [3,8]表示创建一个3*8的矩阵
zero_node = tf.constant(0.)
assign_node = tf.assign(count_variable,zero_node)
sess = tf.Session()
sess.run(assign_node)
print(sess.run(count_variable))'''

from test import Test1




