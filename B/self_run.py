""""
尝试将所有词加入嵌入层训练：无法收敛
"""

import json,keras,numpy
from keras.layers import *
from tqdm import tqdm
from keras.models import load_model, Model
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models import KeyedVectors, Word2Vec, word2vec

w2i = json.load(open('W2C\word_ls.json', encoding='utf8'))
MAXLEN = 35
BATCH_SIZE = 30
EPOCHES = 30
pre_w2v = KeyedVectors.load_word2vec_format('W2C/w2c')
weights = np.zeros((len(w2i),100))

def loadTrainData(test_size):
    ONE_HOT = {'positive': [1, 0, 0], 'neutral': [0, 1, 0], 'negative': [0, 0, 1]}
    # 1.统计所有词
    train_tt = json.load(open('dataC/train_tt.json',encoding='utf8'))
    trainB_tt = json.load(open('dataB/train_tt.json',encoding='utf8'))
    test_tt = json.load(open('dataC/test_tt.json', encoding='utf8'))

    trainX, trainY = [], []
    for s in tqdm(train_tt+trainB_tt+test_tt, ncols=70, ascii=True):
        # 权重
        for w in s['news_comment']:
            try:
                weights[w2i[w]] = pre_w2v[w]
            except:
                pass
        #训练集
        if 'polarity' in s:
            x,y = [],[]
            for w in s['news_comment']:
                x.append(w2i[w])
            trainX.append(x)
            trainY.append(ONE_HOT[s['polarity']])
    print("权重维度：", weights.shape)
    trainX,trainY = np.array(trainX), np.array(trainY)
    trainX = pad_sequences(trainX, MAXLEN)
    return trainX[test_size:],trainY[test_size:],trainX[:test_size],trainY[:test_size]



def genModel():
    X = Input(shape=(MAXLEN,))
    x = Embedding(
        input_dim = len(w2i),
        output_dim = 100,
        input_length = MAXLEN,
        mask_zero=True,
        trainable=True,
        embeddings_initializer = Constant(weights)
    )(X)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.5)(x)
    Y = Dense(3)(x)

    model = Model(inputs=X, outputs=Y)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":

    x_tr,y_tr,x_val,y_val = loadTrainData(1000)

    model = genModel()
    model.fit(
        x_tr,
        y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHES,
        validation_data=(x_val,y_val)
    )
