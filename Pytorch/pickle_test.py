import pickle
import numpy as np


x = np.arange(12.).reshape(2, 2, 3)

# 保存文件
pickle.dump(x, open("pickle.pkl", mode="wb"))

# pickle读文件
y = pickle.load(open("pickle.pkl", mode="rb"))

print(np.all(x == y))   # True
