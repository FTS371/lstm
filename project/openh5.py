from keras.models import load_model
from keras.utils import plot_model
import os
# 加载模型
model = load_model('output/lstm_model.h5')
os.environ["PATH"] += os.pathsep +'D:/Graphviz/bin/'
# 可视化模型结构并保存为png图片
plot_model(model, to_file='output/lstm_model.png', show_shapes=True)