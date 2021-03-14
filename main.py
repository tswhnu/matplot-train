import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE_PATH_1 = './data/original_train_loss.csv'
CSV_FILE_PATH_2 = './data/v0.01train_loss.csv'
df1 = pd.read_csv(CSV_FILE_PATH_1)
df2 = pd.read_csv(CSV_FILE_PATH_2)
x = df1['Step']
y1 = df1['Value']
y2 = df2['Value']
#tttttttttttt
plt.plot(x, y1, label = "original version loss")  # 绘制x,y的折线图
plt.plot(x, y2, label = "v0.01 version loss")  # 绘制x,y的折线图
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid()
plt.savefig("v001_loss")
plt.show()  # 显示折线图
