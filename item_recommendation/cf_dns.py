import tensorflow as tf
from dis_model_dns import DIS
#import cPickle
import pickle
import numpy as np
import multiprocessing # 多进程

cores = multiprocessing.cpu_count()#计算计算机的cpu核心数
print(cores) # 四核
#########################################################################################
# Hyper-parameters
#########################################################################################
# 用户id一直到了942，项目id纸看到了1329，需要核对训练数据集包含多少不同的用户，多少不同的项目
# movieLens 100K 数据集包括 943 个用户对 1682 部电影的100000个电影的评分（1~5）
EMB_DIM = 5
USER_NUM = 943
ITEM_NUM = 1683
DNS_K = 5  # 从候选集中随机选择 5 个作为
all_items = set(range(ITEM_NUM)) #range() 函数可创建一个整数列表，set()返回一个无序不重复元素集
#print(all_items)
workdir = 'ml-100k/'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'
DIS_MODEL_FILE = workdir + "model_dns.pkl"
#########################################################################################
# Load data
#########################################################################################
user_pos_train = {}
#with open(workdir + 'movielens-100k-train.txt')as fin:
with open(workdir + 'movielens-100k-train-gai.txt')as fin: #with open的好处，不需要每次都 f.close()
    for line in fin:
        line = line.split() #通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
        #print(line)
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
               # print("in")
            else:
                user_pos_train[uid] = [iid]
                #print("not")
#print(user_pos_train) # 用户喜欢*哪些项目（评分大于3.99）{0:[],1:[]...}

user_pos_test = {}
with open(workdir + 'movielens-100k-test-gai.txt')as fin:
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

#all_users = user_pos_train.keys() #dict.keys() 2中以列表的形式返回一个字典的所有键值，3返回的不再是一个列表，而是一个可读的view
#all_users.sort() # 用于数组排序
all_users = sorted(user_pos_train)
#print(all_users)
# python 中定义一个函数要使用 def 语句，依次写出函数名、括号、括号中的参数和冒号，return 语句返回
def generate_dns(sess, model, filename):
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]
        all_rating = sess.run(model.dns_rating, {model.u: u})
        all_rating = np.array(all_rating) # 构造函数
        print(all_rating)
        neg = []
        candidates = list(all_items - set(pos)) # 候选集：所有项目 -- 用户喜欢的项目（剩下的是为评分的和评分小于等于3.99的）

        for _ in range(len(pos)):
            choice = np.random.choice(candidates, DNS_K) # 以多大的概率，从候选集中选择 几个 数据
            choice_score = all_rating[choice]
            print(choice_score)
            neg.append(choice[np.argmax(choice_score)]) # .argmax() 取出choice_score 中最大值对应的索引

        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


def dcg_at_k(r, k): # 折扣增益
    r = np.asfarray(r)[:k] # 把一个普通的数组转为一个浮点类型的数组
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k): # 归一化折扣增益
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

    item_score = sorted(item_score, key=lambda x: x[1], reverse=True)
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3]) # 求均值
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
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret


def generate_uniform(filename):
    data = []
    #print 'uniform negative sampling...'
    print('uniform negative sampling...')
    for u in user_pos_train:
        pos = user_pos_train[u]
        candidates = list(all_items - set(pos))
        neg = np.random.choice(candidates, len(pos))
        pos = np.array(pos)

        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


def main():
    np.random.seed(70) # 利用随机数种子，每次生成的随机数相同
    param = None
    discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.1, param=param, initdelta=0.05, learning_rate=0.05)

    config = tf.ConfigProto() # 配置session 运行参数&&GPU设备指定
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) # 初始化模型的参数

    dis_log = open(workdir + 'dis_log_dns.txt', 'w')
    #print "dis ", simple_test(sess, discriminator)
    print("dis",simple_test(sess,discriminator))
    best_p5 = 0.

    # generate_uniform(DIS_TRAIN_FILE) # Uniformly sample negative examples
    
    #for epoch in range(80):
    for epoch in range(10):
        generate_dns(sess, discriminator, DIS_TRAIN_FILE)  # dynamic negative sample
        with open(DIS_TRAIN_FILE)as fin:
            for line in fin:
                line = line.split()
                u = int(line[0])
                i = int(line[1])
                j = int(line[2])
                _ = sess.run(discriminator.d_updates,
                             feed_dict={discriminator.u: [u], discriminator.pos: [i],
                                        discriminator.neg: [j]})

        result = simple_test(sess, discriminator)
        #print "epoch ", epoch, "dis: ", result
        print("epoch",epoch,"dis:",result)
        if result[1] > best_p5:
            best_p5 = result[1]
            discriminator.save_model(sess, DIS_MODEL_FILE)
            #print "best P@5: ", best_p5
            print("best p@5",best_p5)

        buf = '\t'.join([str(x) for x in result])
        dis_log.write(str(epoch) + '\t' + buf + '\n')
        dis_log.flush()

    dis_log.close()


if __name__ == '__main__':
    main()
