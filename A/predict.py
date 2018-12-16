from utils import *
import utils

def __getModel():
    """选择最近训练模型"""
    model_files = os.listdir('./model')
    path = './model/%s'% max(model_files)
    print("选择模型：%s"%path)
    return load_model(path)

def evaluate():
    labels = utils.labels()
    X,Y = feed(labels[-3000:])
    model = __getModel()
    loss, acc = model.evaluate(X,Y)
    print('loss=',loss)
    print('acc=',acc)

def predict_helper(path, model):
    img = cv2.imread(path,0)
    a,b,c,d = preParse(img)
    sample = np.array([a,b,c,d])
    pre = model.predict(sample)
    return ''.join(a2c(pre))

def predict():
    labels = utils.labels()
    pre_codes = []
    real_codes = []
    model = __getModel()
    for sp in tqdm(labels[47000:],ascii=True,ncols=50):
        pred_code = predict_helper(sp[0], model)
        real_code = sp[1]
        if pred_code != real_code:
            print('样本：',sp,'错误预测为：',pred_code)
        pre_codes.append(pred_code)
        real_codes.append(real_code)
    return pre_codes,real_codes

def test_submit():
    test_files = os.listdir('./test')
    model = __getModel()
    submit = []
    for file in tqdm(test_files,ascii=True,ncols=50):
        sample = {
            'id':file,
            'characters':predict_helper('./test/%s'%file, model)
        }
        submit.append(sample)
    print('submit实例个数：',len(submit))
    with open('./submit/prediction.json','w') as f:
        f.write(json.dumps(submit,indent=2))
        print('提交文件保存到./submit/prediction.json')

if __name__ == '__main__':
    # pre, real = predict()
    # score = metrics.accuracy_score(real,pre)
    # print("准确率：", score)
    test_submit()
