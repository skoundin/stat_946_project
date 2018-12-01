# ---------------------------------------------------------------------------------------
# Image Captioning using the Transformer network
# ---------------------------------------------------------------------------------------
from tensorflow.keras.layers import Embedding, TimeDistributed, Dense, Input, Lambda
import keras.backend as K

from external.attention_is_all_you_need.transformer import Transformer, get_pos_encoding_matrix, Decoder
import external.attention_is_all_you_need.dataloader as dd
import external.attention_is_all_you_need.transformer as transformer


if __name__ == '__main__':

    max_caption_length = 70
    dim_embedding = 256
    vocab_size = 5001

    dim_model = 512
    dim_hidden = 512
    num_attention_heads = 8
    dim_key = 64
    dim_value = 64
    num_decoder_layers = 2
    prob_dropout = 0.1

    # Position encoding layer, this is used to preserve the order of words in caption sequences.
    pos_embed_layer = Embedding(
        max_caption_length,
        dim_embedding,
        trainable=False,
        weights=[get_pos_encoding_matrix(max_caption_length, dim_embedding)]
    )

    # Word Embedding
    word_embed_layer = Embedding(vocab_size, dim_embedding)

    # The Full Decoder
    decoder = Decoder(
        d_model=dim_model,
        d_inner_hid=dim_hidden,
        n_head=num_attention_heads,
        d_k=dim_key,
        d_v=dim_value,
        layers=num_decoder_layers,
        dropout=prob_dropout,
        word_emb=word_embed_layer,
        pos_emb=pos_embed_layer)





