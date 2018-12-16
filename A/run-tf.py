"""
tensorflow
"""
from utils import *

TARGET_CLASSES = 52
INPUT_SHAPE = (None, 40,40,1)
OUTPUT_SHAPE = (None,TARGET_CLASSES)
BATCH_SIZE = 64
EPOCHS = 10

x = tf.placeholder(tf.float32,shape = INPUT_SHAPE)
y_ = tf.placeholder(tf.float32,shape = OUTPUT_SHAPE)


cov1 = tf.layers.conv2d(x,32, (3,3), strides=1, padding='same', activation=tf.nn.relu)
cov2 = tf.layers.conv2d(cov1, 64, (3,3), strides=1, padding= 'same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(cov2,(2,2),strides=(1,1), padding='same')
drop1 = tf.layers.dropout(pool1, 0.25)
flatten = tf.layers.flatten(drop1)
dense1 = tf.layers.dense(flatten, 128, activation=tf.nn.relu)
drop2 = tf.layers.dropout(dense1, 0.5)
dense2 = tf.layers.dense(drop2, TARGET_CLASSES, activation=tf.nn.softmax)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=dense2)
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)



labels = labels()
X,Y = feed(labels[:500])
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.14)

if __name__ == '__main__':

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # v = sess.run(optimizer, {x:x_train, y_: y_train})
        # print(v)
        # saver = tf.train.Saver()
        # saver.save(sess=sess, save_path='./test-model')
        for ep in range(EPOCHS):
            index = np.random.permutation(len(x_train))
            for i in tqdm(range(len(x_train)//BATCH_SIZE),ascii=True,ncols=50):
                sess.run(optimizer,{x:x_train[index[BATCH_SIZE*i:BATCH_SIZE*(i+1)]], y_:y_train[index[BATCH_SIZE*i:BATCH_SIZE*(i+1)]]})
            y_test = sess.run(loss,{x:x_test,y_:y_test})
            print(y_test)
            print(a2c(y_test))