import json,os, keras,jieba,time,random
from gensim.models import KeyedVectors,Word2Vec,word2vec
import numpy as np
from tqdm import tqdm
import threading
from keras.models import load_model, Model
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, SimpleRNN, Input, Add
import keras.backend.tensorflow_backend as TF
import tensorflow as tf
from keras.initializers import Constant

EPOCHES = 30
BATCH_SIZE = 30
MAXLEN = 30

wvs = KeyedVectors.load_word2vec_format('W2C/w2c')
words = wvs.index2entity

stop = json.load(open('./dataB/stop.txt','r', encoding='utf8'))

def getW2I():
    w2i = {}
    for i in range(len(words)):
        w2i[words[i]] = i
    return w2i
w2i = getW2I()


def cutWords(src, dst, b1,b2):
    """分词并保存"""
    with open('./dataC/%s' % src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    n_data = []
    for e in data:
        tmp = []
        cut = list(jieba.cut(e['news_comment'], False, True))
        for w in cut:
            if not w in stop:
                tmp.append(w)
        e['news_comment'] = tmp
        n_data.append(e)
    with open('./dataC/%s' % dst, 'w', encoding='utf8') as f:
        f.write(json.dumps(n_data,ensure_ascii=False,indent=2))


def trainWords():
    """读取训练集，由预训练模型分词并生成词向量"""
    with open('./dataB/train_tt.json', mode='r', encoding='utf8') as f:
        train = json.load(f)
    with open('./dataB/test_tt.json', mode='r', encoding='utf8') as f:
        test = json.load(f)
    with open('./dataC/train_tt.json', mode='r', encoding='utf8') as f:
        train_c = json.load(f)
    with open('./dataC/test_tt.json', mode='r', encoding='utf8') as f:
        test_c = json.load(f)
    with open('./dataB/add_data_neg.json', mode='r', encoding='utf8') as f:
        neg = json.load(f)
    with open('./dataB/add_data_pos.json', mode='r', encoding='utf8') as f:
        pos = json.load(f)
    with open('./dataB/add_data_nur.json', mode='r', encoding='utf8') as f:
        nur = json.load(f)
    with open('./dataB/weibo_with_label.json', mode='r', encoding='utf8') as f:
        weibo = json.load(f)
    data = [s['news_comment'] for s in train+train_c]  # 字典转数组
    data += [s['news_comment'] for s in test+test_c]
    data += [s['news_comment'] for s in weibo+neg+pos+nur]
    wvm = Word2Vec(min_count=5)
    wvm.build_vocab(data)
    wvm.train(data, epochs=30, total_examples=wvm.corpus_count,
              total_words=wvm.corpus_total_words)
    wvm.wv.save_word2vec_format('W2C/w2c')
    print('词向量模型已保存')
    return wvm


def loadTrainData(test_num, num_sp = None):
    """读取训练集，由预训练模型分词并生成词向量"""
    ONE_HOT = {'positive':[1,0,0],'neutral':[0,1,0],'negative':[0,0,1]}


    with open('./dataB/train_tt.json',mode='r', encoding='utf8') as f:
        dataB = json.load(f)
    with open('./dataC/train_tt.json', mode='r', encoding='utf8') as f:
        dataC = json.load(f)

    data = dataB+dataC
    del dataB, dataC
    if num_sp:
        data = data[:num_sp]
    random.shuffle(data)
    train = [[s['news_comment'],s['polarity']] for s in data[test_num:]] # 字典转数组
    test = [[s['news_comment'],s['polarity']] for s in data[:test_num]]

    print('---加载训练集---')
    x_train,y_train = [],[]
    for i in tqdm(range(len(train)),ncols=100,ascii=True):
        doc_vecs = []
        for w in train[i][0]:
            if w in w2i:
                doc_vecs.append(w2i[w])
        x_train.append(doc_vecs)
        y_train.append(ONE_HOT[train[i][1]])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)

    print('---加载验证集---')
    x_test,y_test = [],[]
    for i in tqdm(range(len(test)),ncols=100,ascii=True):
        doc_vecs = []
        for w in test[i][0]:
            if w in w2i:
                doc_vecs.append(w2i[w])
        x_test.append(doc_vecs)
        y_test.append(ONE_HOT[test[i][1]])

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)
    return x_train,y_train, x_test,y_test


def initModel():
    model = keras.Sequential()
    model.add(Embedding(
        input_dim = len(w2i),
        output_dim = wvs['但是'].shape[0], 
        input_length = MAXLEN, 
        mask_zero = False,
        trainable = False,
        weights=[np.array([wvs[w] for w in wvs.index2entity])]
        )
    )
    model.add(Bidirectional(LSTM(60)))
    model.add(Dropout(0.6))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy']) 
    return model

def cmp_model():
    inp = Input(shape=(MAXLEN,))
    em = Embedding(
        input_dim=len(w2i),
        output_dim=wvs['但是'].shape[0],
        input_length=MAXLEN,
        mask_zero=True,
        trainable=True,
        embeddings_initializer=Constant(wvs.vectors)
    )(inp)
    x = Dropout(0.2)(em)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.50)(x)
    
    out = Dense(3,activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def predict(model, data = None):
    """预测测试集"""
    ONE_HOT = ['positive', 'neutral', 'negative']
    # model = keras.models.load_model('%s.h5'%model_name)

    if not data:
        with open('./dataC/test_tt.json', 'r', encoding='utf-8')  as f:
            data = json.loads(f.read())
    
    answers, ids, vecs = [], [], []
    for s in tqdm(data, ascii=True, ncols=50):
        ids.append(s['id'])
        vecs.append([w2i[w] for w in s['news_comment'] if w in w2i])
    
    vecs = np.array(sequence.pad_sequences(vecs,maxlen=MAXLEN))

    print('----数据加载完成，开始预测----')
    polarities = model.predict(vecs)

    # print(polarities)
    for i in range(len(ids)):
        sample = {}
        sample['id'] = ids[i]
        polarity = polarities[i]
        sample['polarity'] = ONE_HOT[list(polarity).index(max(polarity))]
        answers.append(sample)
    with open('submitC/prediction_new2.json', 'w') as f:
        f.write(json.dumps(answers,indent=2))


if __name__ == "__main__":
    # trainWords()

    # cutWords('test.json', 'test_tt.json', True, True)
    # cutWords('train.json', 'train_tt.json', True, True)

    # x_train, y_train, x_test,y_test = loadTrainData(1400)
    # model = cmp_model()
    # acc_save = 0.72
    # for ep in range(EPOCHES):
    #     tr_loss,tr_acc = 0,0
    #     index = np.random.permutation(len(x_train))
    #     for i in tqdm(range(len(x_train)//BATCH_SIZE),ascii=True,ncols=50):
    #         t_loss,t_acc = model.train_on_batch(x_train[index[BATCH_SIZE*i:BATCH_SIZE*(i+1)]], y_train[index[BATCH_SIZE*i:BATCH_SIZE*(i+1)]])
    #         tr_loss += t_loss
    #         tr_acc += t_acc
    #     tr_loss /= len(x_train)//BATCH_SIZE
    #     tr_acc /= len(x_train)//BATCH_SIZE
    #     loss, acc = model.evaluate(x_test,y_test,batch_size=BATCH_SIZE,verbose=0)
    #     print('step=%d:  loss=%.6f  acc=%.6f  vali_loss=%.6f  vali_acc=%.6f'%(ep,tr_loss,tr_acc,loss,acc))

    #     if acc>acc_save:
    #         print('~~~~~~~~~~~~~')
    #         acc_save = acc
    #         model.save('./model_test/model-%s-%.6f.h5' % (time.strftime('%d%H%M%S'),acc))
    #     if tr_acc-acc > 0.1:
    #         break


    # model.fit(x_train,y_train,
    #     batch_size=BATCH_SIZE,
    #     epochs=EPOCHES,
    #     verbose=1,
    #     validation_data=(x_test, y_test)
    # )
    # # model.save('./model_test/add_label_f.h5')

    model = load_model('model_test\model-15010110-0.740714.h5')
    # a = model.predict(x_test)
    # count = 0
    # for i in range(len(a)):
    #     if list(a[i]).index(max(a[i])) != list(y_test[i]).index(1):
    #         count += 1
    #         print(np.around(a[i],2))
    #         print(y_test[i])
    # print('错误数量：', count)


    predict(model)
