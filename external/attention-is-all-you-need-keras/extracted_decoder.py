
import numpy
import matplotlib.pyplot as plt

from keras.layers import Embedding, TimeDistributed, Dense, Input, Lambda
import keras.backend as K

import transformer
import dataloader as dd


if __name__ == '__main__':
    plt.ion()

    # Positional Embedding
    pos_emb = Embedding(
        input_dim=70,
        output_dim=512,
        trainable=False,
        weights=[transformer.GetPosEncodingMatrix(70, 512)])

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    # Word Embedding
    i_tokens, o_tokens = dd.MakeS2SDict('data/en2de.s2s.txt', dict_file='data/en2de_word.txt')
    o_word_emb = Embedding(o_tokens.num(), 512)






    decoder = transformer.Decoder(
        d_model=512,
        d_inner_hid=512,
        n_head=8,
        d_k=64,
        d_v=64,
        layers=2,
        dropout=0.1,
        word_emb=None,
        pos_emb=pos_emb)

    target_layer = TimeDistributed(Dense(o_tokens.num(), use_bias=False))

    src_seq_input = Input(shape=(None,), dtype='int32')
    tgt_seq_input = Input(shape=(None,), dtype='int32')

    src_seq = src_seq_input
    tgt_seq = Lambda(lambda x: x[:, : -1])(tgt_seq_input)
    tgt_true = Lambda(lambda x: x[:, 1:])(tgt_seq_input)

    src_pos = Lambda(self.get_pos_seq)(src_seq)
    tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
    if not self.src_loc_info:
        src_pos = None

    enc_output = self.encoder(src_seq, src_pos, active_layers=active_layers)
    dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, active_layers=active_layers)
    final_output = self.target_layer(dec_output)





    dec_output = decoder(tgt_seq, tgt_pos, src_seq, enc_output, active_layers=active_layers)
    final_output = self.target_layer(dec_output)