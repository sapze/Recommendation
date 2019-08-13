# -*- coding: utf-8 -*-
import tensorflow as tf
'''from dis_model import DIS
from gen_model import GEN'''
from dis_gai import DIS
from gen_gai import GEN
import pickle
import numpy as np
import utils as ut
import multiprocessing

print("test")
cores = multiprocessing.cpu_count()

# Hyper-parameters
EMB_DIM = 5
#USER_NUM = 6
#ITEM_NUM = 457
RELATION_NUM = 7
USER_NUM = 943
ITEM_NUM = 1683
BATCH_SIZE = 16
INIT_DELTA = 0.05

all_items = set(range(ITEM_NUM))
workdir = 'ml-100k/'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'

#########################################################################################
# Load data
#########################################################################################
user_pos_train = {}
with open(workdir + 'movielens-100k-train.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]

user_pos_test = {}
with open(workdir + 'movielens-100k-test.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]

#all_users = user_pos_train.keys()
#all_users.sort()
all_users = sorted(user_pos_train.keys())
print(len(all_users))


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]  # 不会占用新的内存
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def simple_test_one_user(x):
    rating = x[0]
    u = x[1]

    test_items = list(all_items - set(user_pos_train[u]))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])   # 按照评分排序
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)    # r 指示预测项目是否与真实用户喜欢的项目相一致

    p_3 = np.mean(r[:3])   # r[:3]  r[0]\r[1]\r[2]
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])


def simple_test(sess, model):
    result = np.array([0.] * 6)
    pool = multiprocessing.Pool(cores)
    batch_size = 2
    #batch_size = 128
    #test_users = user_pos_test.keys()
    test_users = list(user_pos_test.keys())
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)  # python2将可迭代的对象作为参数,将对象中对应的元素打包成一个个元组,然后返回由这些元组组成的列表 #接受任意多个可迭代对象作为参数,将对象中对应的元素打包成一个tuple,然后返回一个可迭代的zip对象.
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)  # 第一个参数函数，第二个参数需要迭代的列表，把第二个从参数不断的传到第一个函数的参数中
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret


def generate_for_d(sess, model, filename):  # 为鉴别器生成数据
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]

        rating = sess.run(model.all_rating, {model.u: [u]})
        rating = np.array(rating[0]) / 0.2  # Temperature
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)
        
        #print(prob)
        #print(np.sum(prob))

        neg = np.random.choice(np.arange(ITEM_NUM), size=len(pos), p=prob) # 以p的概率从第一个参数中选取size个数
        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


def main():
    print("load model...")
    param = None  # 相当于预训练，param是否指定参数
    #param = pickle.load(open(workdir + "model_dns_ori.pkl"))
    #param = cPickle.load(open(workdir + "model_dns_ori.pkl"))
    generator = GEN(ITEM_NUM, USER_NUM, RELATION_NUM, EMB_DIM, lamda=0.0 / BATCH_SIZE, param=param, initdelta=INIT_DELTA,
                    learning_rate=0.001)
    discriminator = DIS(ITEM_NUM, USER_NUM, RELATION_NUM, EMB_DIM, lamda=0.1 / BATCH_SIZE, param=None, initdelta=INIT_DELTA,
                        learning_rate=0.001)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())  
    # 训练自己的神经网络的时候，无一例外的就是都会加上一句 sess.run(tf.global_variables_initializer()) ，这行代码的官方解释是 初始化模型的参数

    print ("gen ", simple_test(sess, generator))
    print ("dis ", simple_test(sess, discriminator))

    dis_log = open(workdir + 'dis_log.txt', 'w')
    gen_log = open(workdir + 'gen_log.txt', 'w')

    # minimax training
    best = 0.
    for epoch in range(5):  # 15
        if epoch >= 0:
            for d_epoch in range(3):  # 100 # Train D
                if d_epoch % 5 == 0:
                    generate_for_d(sess, generator, DIS_TRAIN_FILE)
                    train_size = ut.file_len(DIS_TRAIN_FILE)
                index = 1
                while True:
                    if index > train_size:
                        break
                    if index + BATCH_SIZE <= train_size + 1:
                        input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index, BATCH_SIZE)
                    else:
                        input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index,
                                                                                train_size - index + 1)
                    index += BATCH_SIZE

                    '''_ = sess.run(discriminator.d_updates,
                                 feed_dict={discriminator.u: input_user, discriminator.i: input_item,
                                            discriminator.label: input_label})'''
                    _ = sess.run(discriminator.d_updates,
                                 feed_dict={discriminator.u: input_user, discriminator.i: input_item,
                                            discriminator.label: input_label,discriminator.r: 1})

            # Train G
            for g_epoch in range(3):  # 50
                for u in user_pos_train:
                    sample_lambda = 0.2
                    pos = user_pos_train[u]

                    rating = sess.run(generator.all_logits, {generator.u: u})
                    exp_rating = np.exp(rating)
                    prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta

                    pn = (1 - sample_lambda) * prob
                    pn[pos] += sample_lambda * 1.0 / len(pos)
                    # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta
                    #print(prob,np.sum(prob))
                    #print(pn)
                    #print(np.sum(pn))
                    sample = np.random.choice(np.arange(ITEM_NUM), 2 * len(pos), p=pn)
                    #print(len(sample))
                    ###########################################################################
                    # Get reward and adapt it with importance sampling
                    ###########################################################################
                    reward = sess.run(discriminator.reward, {discriminator.u: u, discriminator.i: sample})
                    reward = reward * prob[sample] / pn[sample]
                    ###########################################################################
                    # Update G
                    ###########################################################################
                    '''_ = sess.run(generator.gan_updates,
                                 {generator.u: u, generator.i: sample, generator.reward: reward})'''
                    _ = sess.run(generator.gan_updates,
                                 {generator.u: u, generator.i: sample, generator.r:1, generator.reward: reward})

                result = simple_test(sess, generator)
                print ("epoch ", epoch, "gen: ", result)
                buf = '\t'.join([str(x) for x in result])
                gen_log.write(str(epoch) + '\t' + buf + '\n')
                gen_log.flush()  # 用来刷新缓冲区，将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入

                p_5 = result[1]
                if p_5 > best:
                    print( 'best: ', result)
                    best = p_5
                    generator.save_model(sess, "ml-100k/gan_generator.pkl")

    gen_log.close()
    dis_log.close()


if __name__ == '__main__':
    main()

