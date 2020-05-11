"""
@author: LiShiHang
@software: PyCharm
@file: remove_top.py
@time: 2019/8/14 11:33
@desc:r
"""
import keras

base = keras.models.load_model(r"C:\Users\shihangli\Desktop\rs_pretrain_model_rscup2019-densenet121.h5")
model = keras.models.Model(inputs=base.input,outputs=base.layers[-3].output)

model.summary()

model.save("output/rs_pretrain_model_rscup2019-densenet121-notop.h5")