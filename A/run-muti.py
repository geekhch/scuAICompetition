from utils import *
import predict

# 参数设置
#############################################
INPUT_SHAPE = (40, 240, 1)
NUM_CLASSES = 52
BATCH_SIZE = 64
EPOCHS = 24

def model():
    """共享层实现多输出网络，端对端直接输出四个验证码字符"""
    inputs = Input(shape=INPUT_SHAPE)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x1 = Dense(NUM_CLASSES, activation='softmax',name='x1')(x)
    x2 = Dense(NUM_CLASSES, activation='softmax',name='x2')(x)
    x3 = Dense(NUM_CLASSES, activation='softmax',name='x3')(x)
    x4 = Dense(NUM_CLASSES, activation='softmax',name='x4')(x)

    model = Model(inputs=inputs, outputs=[x1,x2,x3,x4])
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':

    model = model()

    labels = labels()
    X,Y = feed_muti(labels[:30000])
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

    print('shape of x_train:',x_train.shape)
    model.fit(x_train,
            {
                'x1':y_train[:,0,:],
                'x2':y_train[:,1,:],
                'x3':y_train[:,2,:],
                'x4':y_train[:,3,:]
            },
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(x_test, {
                'x1':y_test[:,0,:],
                'x2':y_test[:,1,:],
                'x3':y_test[:,2,:],
                'x4':y_test[:,3,:]
            }))
    model.save('./model/muti-%s.h5'%time.strftime('%m-%d-%H%M%S'))
    

