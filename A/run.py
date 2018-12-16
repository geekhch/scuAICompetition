from utils import *
import predict

# TF.set_session(tf.Session(config = tf.ConfigProto(device_count={'gpu':0})))

# 参数设置
#############################################
INPUT_SHAPE = (40, 40, 1)
NUM_CLASSES = 52
BATCH_SIZE = 100
EPOCHS = 40



def model():
    model = Sequential()
    model.add(Conv2D(40, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=INPUT_SHAPE))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(156, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy']) 
    return model


if __name__ == '__main__':

    model = model()

    labels = labels()
    X,Y = feed(labels[:50000])
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.16)
    del X,Y
    # model.fit(x_train, y_train,
    #     batch_size=BATCH_SIZE,
    #     epochs=EPOCHS,
    #     verbose=1,
    #     validation_data=(x_test, y_test))
    # model.save('./model/%s.h5'%time.strftime('%m-%d-%H%M%S'))

    # pre, real = predict.predict()
    # score = metrics.accuracy_score(real,pre)
    # print("准确率：", score)

    # # 画模型图
    # plot_model(model, to_file='model.png')
    #############################################3
    for ep in range(EPOCHS):
        index = np.random.permutation(len(x_train))
        for i in tqdm(range(len(x_train)//BATCH_SIZE),ascii=True,ncols=50):
            tr_loss,tr_acc = model.train_on_batch(x_train[index[BATCH_SIZE*i:BATCH_SIZE*(i+1)]], y_train[index[BATCH_SIZE*i:BATCH_SIZE*(i+1)]])
        loss, acc = model.evaluate(x_test,y_test,batch_size=BATCH_SIZE,verbose=1)
        print('    step=%d:  loss=%.6f  acc=%.6f  vali_loss=%.6f  vali_acc=%.6f'%(ep,tr_loss,tr_acc,loss,acc))
        if acc > 0.99994:
            model.save('./model/%s-batch-%s-%d.h5'%(time.strftime('%m-%d-%H%M%S'),str(round(acc,6)),ep))