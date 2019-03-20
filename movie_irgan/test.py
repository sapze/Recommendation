# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle 
import utils as ut
import multiprocessing  # 多进程
from gen_model import GEN
from dis_model import DIS

cores = multiprocessing.cpu_count() # 计算cpu核数
print(cores)

# 温度参数设置为 0.2
EMB_DIM = 5     # 矩阵分解中的因子数目，潜在维度
USER_NUM = 943  # 用户数目
ITEM_NUM = 1683 # 项目数目
BATCH_SIZE = 16
INIT_DELTA = 0.05

all_items = set(range(ITEM_NUM)) 
workdir = 'ml-100k/'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'

# 采用NetRank两层神经网络来计算查询和文档之间的相关分数
# 采用矩阵分解来为用户项目偏好打分
# 采用余弦相似度来衡量问题和回答之间的相似性

#########################################################################################
# Load data
#########################################################################################
# 采用的是隐性反馈数据，将 5 星评级作为正反馈,其它作为未知反馈
user_pos_train = {}
with open(workdir + 'movielens-100k-train.txt')as fin:
    for line in fin:
        line = line.split() #通过指定分隔符对字符串进行切片
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:   # 是否应该为4.99?
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

all_users = sorted(user_pos_train) # 先求出user_pos_train 字典型变量的键值，然后对其排序 # 得到用户从小到大的排序[0,1,2...]
print(all_users)
# 最终要的是生成模型生成的能够以假乱真的内容
# 计算 DCG 折扣增益
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

# 计算归一化折扣增益
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

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()     # list.reverse() 对列表的元素进行反向排序
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])

def simple_test(sess, model):
    result = np.array([0.] * 6)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
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
        user_batch_rating_uid = zip(user_batch_rating, user_batch) 
        # 将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
        # 我们可以使用 list() 转换来输出列表。
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        # 进程池的使用是为了提高效率，第一个参数是一个函数，第二个参数是一个迭代器，将迭代器中的数字作为参数依次传入函数中
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret


def generate_for_d(sess, model, filename):  # 生成器生成文档
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]   # 求得每一个用户的积极样本

        rating = sess.run(model.all_rating, {model.u: [u]})
        rating = np.array(rating[0]) / 0.2  # Temperature
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)

        neg = np.random.choice(np.arange(ITEM_NUM), size=len(pos), p=prob)
        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


def main():
    print("load model...")
    #param = pickle.load(open(workdir + "model_dns_ori.pkl"))   #.pkl是python 用来保存文件的
    with open(workdir + "model_dns_ori.pkl",'rb') as data_file:
        param = pickle.load(data_file,encoding='bytes') 
    #param = cPickle.load(open(workdir + "model_dns_ori.pkl"))
    #with open(workdir + "model_dns_ori.pkl",'rb') as data_file:
        #param = pickle.load(data_file,encoding='bytes')
    print(param)
    generator = GEN(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.0 / BATCH_SIZE, param=param, initdelta=INIT_DELTA,
                    learning_rate=0.001)
    discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.1 / BATCH_SIZE, param=None, initdelta=INIT_DELTA,
                        learning_rate=0.001)

    config = tf.ConfigProto()  # 一般用在创建session的时候。用来对session进行参数配置，配置session运行参数&&GPU设备指定
    config.gpu_options.allow_growth = True ## 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
                                           #内存，所以会导致碎片
    sess = tf.Session(config=config) # 要运行刚才定义的三个操作中的任何一个，我们需要为Graph创建一个Session。 Session还将分配内存来存储变量的当前值
    sess.run(tf.global_variables_initializer())

    print("gen ", simple_test(sess, generator))
    print("dis ", simple_test(sess, discriminator))

    dis_log = open(workdir + 'dis_log.txt', 'w')
    gen_log = open(workdir + 'gen_log.txt', 'w')

    # minimax training
    best = 0.
    for epoch in range(15):
        if epoch >= 0:
            for d_epoch in range(100):
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

                    _ = sess.run(discriminator.d_updates,
                                 feed_dict={discriminator.u: input_user, discriminator.i: input_item,
                                            discriminator.label: input_label})

            # Train G
            for g_epoch in range(50):  # 50
                for u in user_pos_train:
                    sample_lambda = 0.2
                    pos = user_pos_train[u]

                    rating = sess.run(generator.all_logits, {generator.u: u})
                    exp_rating = np.exp(rating)
                    prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta

                    pn = (1 - sample_lambda) * prob
                    pn[pos] += sample_lambda * 1.0 / len(pos)
                    # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta

                    sample = np.random.choice(np.arange(ITEM_NUM), 2 * len(pos), p=pn)
                    ###########################################################################
                    # Get reward and adapt it with importance sampling
                    ###########################################################################
                    reward = sess.run(discriminator.reward, {discriminator.u: u, discriminator.i: sample})
                    reward = reward * prob[sample] / pn[sample]
                    ###########################################################################
                    # Update G
                    ###########################################################################
                    _ = sess.run(generator.gan_updates,
                                 {generator.u: u, generator.i: sample, generator.reward: reward})

                result = simple_test(sess, generator)
                print("epoch ", epoch, "gen: ", result)
                buf = '\t'.join([str(x) for x in result])
                gen_log.write(str(epoch) + '\t' + buf + '\n')
                gen_log.flush()

                p_5 = result[1]
                if p_5 > best:
                    print('best: ', result)
                    best = p_5
                    generator.save_model(sess, "ml-100k/gan_generator.pkl")

    gen_log.close()
    dis_log.close()


if __name__ == '__main__':
    #__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()