import pickle

# pickle读文件
pickle.load(open('./1.3 循环神经网络/1.3.2 文本情感分类/model/ws.pkl', 'rb'))



# 保存文件
pickle.dump(对象, open('./1.3 循环神经网络/1.3.2 文本情感分类/model/ws.pkl', 'wb'))