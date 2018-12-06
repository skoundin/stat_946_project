# -*- coding: utf-8 -*-
# !/usr/bin/python3

import os
import sys
from datetime import datetime

from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import plot_model

import dataloader as dd
from transformer import Transformer, LRSchedulerPerStep, LRSchedulerPerEpoch


BASE_RESULTS_DIR = 'models'
if not os.path.exists(BASE_RESULTS_DIR):
    os.mkdir(BASE_RESULTS_DIR)

itokens, otokens = dd.MakeS2SDict('data/en2de.s2s.txt', dict_file='data/en2de_word.txt')
Xtrain, Ytrain = dd.MakeS2SData('data/en2de.s2s.txt', itokens, otokens, h5_file='data/en2de.h5')
Xvalid, Yvalid = dd.MakeS2SData('data/en2de.s2s.valid.txt', itokens, otokens, h5_file='data/en2de.valid.h5')

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())
print('train shapes:', Xtrain.shape, Ytrain.shape)
print('valid shapes:', Xvalid.shape, Yvalid.shape)

'''
from rnn_s2s import RNNSeq2Seq
s2s = RNNSeq2Seq(itokens,otokens, 256)
s2s.compile('rmsprop')
s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, validation_data=([Xvalid, Yvalid], None))
'''

d_model = 512
s2s = Transformer(
    itokens,
    otokens,
    len_limit=70,
    d_model=d_model,
    d_inner_hid=512,
    n_head=8,
    d_k=64,
    d_v=64,
    layers=2,
    dropout=0.1)

model_save_file = os.path.join(BASE_RESULTS_DIR, 'en2de.model.h5')

lr_scheduler = LRSchedulerPerStep(d_model, 4000)   # there is a warning that it is slow, however, it's ok.
# lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
model_saver = ModelCheckpoint(model_save_file, save_best_only=True, save_weights_only=True)

s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
s2s.model.summary()

print("Saving Model Architecture")
plot_model(s2s.model, to_file=os.path.join(BASE_RESULTS_DIR, 'model.png'), show_shapes=True)

try:
    s2s.model.load_weights(model_save_file)
except:
    print('\n\nnew model')

if 'test' in sys.argv:
    print(s2s.decode_sequence_fast('A black dog eats food .'.split(), delimiter=' '))
    while True:
        quest = input('> ')
        print(s2s.decode_sequence_fast(quest.split(), delimiter=' '))
        # rets = s2s.beam_search(quest.split(), delimiter=' ')
        # for x, y in rets:
        #     print(x, y)
else:
    print("Training Model ...")
    start_time = datetime.now()

    s2s.model.fit(
        [Xtrain, Ytrain],
        None,
        batch_size=64,
        epochs=30,
        validation_data=([Xvalid, Yvalid], None),
        callbacks=[lr_scheduler, model_saver]
    )

    print("Training took {}".format(datetime.now() - start_time))
