# ---------------------------------------------------------------------------------------
# Image Captioning using the Transformer network
# ---------------------------------------------------------------------------------------
from keras.layers import Embedding, TimeDistributed, Dense, Input, Lambda
import keras.backend as K

from external.attention_is_all_you_need.transformer import Transformer
import external.attention_is_all_you_need.dataloader as dd
import external.attention_is_all_you_need.transformer as transformer


if __name__ == '__main__':

    # Positional Embedding
    pos_emb = Embedding(
        input_dim=70,
        output_dim=512,
        trainable=False,
        weights=[transformer.GetPosEncodingMatrix(70, 512)])


    def get_pos_seq(x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask


    # Word Embedding
    i_tokens, o_tokens = dd.MakeS2SDict('data/en2de.s2s.txt', dict_file='data/en2de_word.txt')
    o_word_emb = Embedding(o_tokens.num(), 512)
