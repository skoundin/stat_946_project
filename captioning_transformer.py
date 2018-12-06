# ---------------------------------------------------------------------------------------
# Image Captioning using the Transformer network
# ---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from datetime import datetime
import tqdm
import numpy as np
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Embedding, TimeDistributed, Dense, Input, Lambda
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Model


from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *

from external.attention_is_all_you_need.transformer import Transformer, get_pos_encoding_matrix, Decoder

from image_captioning_with_attention import CnnEncoder, get_mscoco_data, calc_max_length, train_test_split, load_image


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_key_dict, b_size=32, feature_size=(64, 2048), max_caption_len=70, shuffle=True):
        """
        A Python generator (actually a keras sequencer object) that can be used to
        dynamically load images when the batch is run. Saves a lot on memory.

        Compared to a generator a sequencer object iterates over all images once during
        an epoch

        Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

        :param data_key_dict: dictionary of (image_location: caption) of all images in the data set.
        :param b_size: batch_size
        :param feature_size:
        :param shuffle: [default=True]

        """
        self.shuffle = shuffle
        self.feature_size = feature_size
        self.batch_size = b_size
        self.data_key_dict = data_key_dict
        self.max_caption_len = max_caption_len

        self.list_ids = list(self.data_key_dict)
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Routines to run at the end of each epoch

        :return:
        """
        self.idx_arr = np.arange(len(self.data_key_dict))

        if self.shuffle:
            np.random.shuffle(self.idx_arr)

    def __data_generation(self, list_ids_temp):
        """

        :param list_ids_temp:
        :return:
        """
        x_arr = np.zeros((self.batch_size, self.feature_size[0], self.feature_size[1]), dtype='float32')
        y_arr = np.zeros((self.batch_size, self.max_caption_len),  dtype='int32')

        # print("Loading a new batch")

        for img_idx, list_id in enumerate(list_ids_temp):

            temp = np.load(img_name_train[img_idx] + '.npy')

            x_arr[img_idx, ] = temp
            y_arr[img_idx, ] = self.data_key_dict[list_id]

        return [x_arr, y_arr], None

    def __len__(self):
        """ The number of batches per epoch"""
        return int(np.floor(len(self.data_key_dict) / self.batch_size))

    def __getitem__(self, index):
        """
        Get one batch of data
        :param index:
        :return:
        """
        idx_arr = self.idx_arr[index * self.batch_size: (index + 1) * self.batch_size]

        # find the list of ids
        list_ids_temp = [self.list_ids[k] for k in idx_arr]

        x_arr, y_arr = self.__data_generation(list_ids_temp)

        return x_arr, y_arr


class CaptionTransformer:
    def __init__(
            self, d_embed, d_model, d_hidden, n_attn_heads, d_k, d_v, n_decoder_layers, p_dropout, max_capt_len,
            v_size):

        self.encoder = CnnEncoder(d_embed)

        # Position encoding layer, this is used to preserve the order of words in caption sequences.
        self.pos_embed_layer = Embedding(
            max_capt_len,
            d_embed,
            trainable=False,
            weights=[get_pos_encoding_matrix(max_capt_len, d_embed)],
            name='position_embed'
        )

        # Word Embedding Layer
        self.word_embed_layer = Embedding(v_size, d_embed, name ='word_embed')

        self.decoder = Decoder(
            d_model=d_model,
            d_inner_hid=d_hidden,
            n_head=n_attn_heads,
            d_k=d_k,
            d_v=d_v,
            layers=n_decoder_layers,
            dropout=p_dropout,
            word_emb=self.word_embed_layer,
            pos_emb=self.pos_embed_layer)

        self.map_to_word_tokens_layer = TimeDistributed(Dense(v_size, use_bias=False))

    @staticmethod
    def get_pos_seq(x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = tf.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def compile(self, optimizer='adam', active_layers=999):

        # This is for Teacher Forcing
        img_features = Input(shape=(64, 2048,), dtype='float32', name='image_features')
        tgt_seq_input = Input(shape=(None,), dtype='int32', name='true_caption')

        tgt_seq = Lambda(lambda x1: x1[:, :-1], name='tgt_seq')(tgt_seq_input)
        tgt_true = Lambda(lambda x1: x1[:, 1:], name='tgt_true')(tgt_seq_input)

        # tgt_seq = [43, 10, 20, 45, 0, 0, 0, 0]  # tokenized words (sequence)
        # tgt_pos = [ 1,  2,  3,  4, 0, 0, 0, 0]
        tgt_pos = Lambda(self.get_pos_seq, name='map_tgt_seq_to_pos_vector')(tgt_seq)

        enc_output = self.encoder(img_features)
        dec_output = self.decoder(tgt_seq, tgt_pos, None, enc_output, active_layers=active_layers)
        final_output = self.map_to_word_tokens_layer(dec_output)

        def get_loss(args):
            y_pred, y_true = args
            y_true = tf.cast(y_true, 'int32')
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss1 = tf.reduce_sum(loss1 * mask, -1) / tf.reduce_sum(mask, -1)
            loss1 = K.mean(loss1)
            return loss1

        def get_accuracy(args):
            y_pred, y_true = args
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)

        loss = Lambda(get_loss, name='get_loss')([final_output, tgt_true])

        self.ppl = Lambda(K.exp, name='exponential_loss')(loss)
        self.accu = Lambda(get_accuracy, name='get_accuracy')([final_output, tgt_true])

        self.model = Model([img_features, tgt_seq_input], loss)
        self.model.add_loss([loss])

        # TODO: What is this for
        self.output_model = Model([img_features, tgt_seq_input], final_output)

        self.model.compile(optimizer, None)
        self.model.metrics_names.append('ppl')
        self.model.metrics_tensors.append(self.ppl)
        self.model.metrics_names.append('accu')
        self.model.metrics_tensors.append(self.accu)


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()

    # -----------------------------------------------------------------------------------
    # Get Data
    # -----------------------------------------------------------------------------------
    print("Getting Data ...")
    train_captions, img_name_vector = get_mscoco_data()

    # Don't include the top, we only need the features from the last layer, not the classifier
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # TODO: Assume Features are extracted
    # # Store Features of Images and use these for training
    # # --------------------------------------------
    # print("Extracting Features from dataset images ...")
    #
    # # Get unique images, there are multiple captions per image. We only need to store
    # # features of unique images
    # IMAGE_CACHING_BATCH_SIZE = 16
    #
    # encode_train = sorted(set(img_name_vector))
    # image_dataset = tf.data.Dataset.from_tensor_slices(encode_train).map(load_image).batch(IMAGE_CACHING_BATCH_SIZE)
    #
    # start_feature_extract = datetime.now()
    # for img, path in tqdm(image_dataset):
    #
    #     batch_features = image_features_extract_model(img)
    #     batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
    #
    #     for bf, p in zip(batch_features, path):
    #         path_of_feature = p.numpy().decode("utf-8")
    #         np.save(path_of_feature, bf.numpy())
    #
    # print("Image Feature Extracting Step took {}".format(datetime.now() - start_feature_extract))

    # -----------------------------------------------------------------------------------
    # Caption Preprocessing
    # -----------------------------------------------------------------------------------
    # 1. Tokenize the captions (e.g., by splitting on spaces) to generate Vocabulary
    # 2. Limit Vocabulary to save memory, all other words replaced with token "UNK"
    # 3. Create word -> index mapping (to easily translate between them)
    # 4. Pad all captions to same (longest) length
    print("Tokenizing Captions...")

    VOCAB_LENGTH = 5000

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=VOCAB_LENGTH,
        oov_token="<unk>",
        filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(train_captions)

    # Represent each word by its token index
    # Eg. '<start> A skateboarder performing a trick on a skateboard ramp. <end>' ==>
    # [3, 2, 351, 687, 2, 280, 5, 2, 84, 339, 4]
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Mapping word -> index
    tokenizer.word_index = \
        {key: value for key, value in tokenizer.word_index.items() if value <= VOCAB_LENGTH}

    # Putting <unk> token in the word2idx dictionary
    tokenizer.word_index[tokenizer.oov_token] = VOCAB_LENGTH + 1
    tokenizer.word_index['<pad>'] = 0

    # # creating the tokenized vectors
    # train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Map index -> word
    index_word = {value: key for key, value in tokenizer.word_index.items()}

    # padding each vector to the max_length of the captions
    # array([  3,  29,  53,  19,  89, 202, 100,  92,   5,   7, 218,   4,   0,
    #          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    #          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    #          0,   0,   0,   0,   0,   0,   0,   0,   0,   0], dtype=int32)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # calculating the max_length
    # used to store the attention weights
    max_length = calc_max_length(train_seqs)

    # -----------------------------------------------------------------------------------
    # Train Validation Split
    # -----------------------------------------------------------------------------------
    print("Creating Training and Validation Split ... ")

    TRAIN_VALIDATION_SPLIT = 0.2

    img_name_train, img_name_val, cap_train, cap_val = train_test_split(
        img_name_vector,
        cap_vector,
        test_size=TRAIN_VALIDATION_SPLIT,
        random_state=0)

    print("Training ({} images, {} captions). Validation ({} images, {} captions)".format(
        len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)))

    # -----------------------------------------------------------------------------------
    # Create Data set Generators
    # -----------------------------------------------------------------------------------
    print("Creating a Training and validation Datasets ...")
    extracted_img_feature_dim = (64, 2048)
    batch_size = 64

    train_data_dict = {}
    for idx in range(len(img_name_train)):
        train_data_dict[img_name_train[idx]] = cap_train[idx]

    train_data_generator = DataGenerator(
        train_data_dict,
        b_size=batch_size,
        shuffle=True,
        max_caption_len=max_length,
        feature_size=extracted_img_feature_dim
    )

    # gen_out = iter(train_data_generator)
    # train_images, train_labels = gen_out.__next__()

    valid_data_dict = {}
    for idx in range(len(img_name_val)):
        valid_data_dict[img_name_val[idx]] = cap_val[idx]

    val_data_generator = DataGenerator(
        valid_data_dict,
        b_size=batch_size,
        shuffle=True,
        max_caption_len=max_length,
        feature_size=extracted_img_feature_dim
    )

    gen_out = iter(val_data_generator)
    val_images, val_labels = gen_out.__next__()

    # -----------------------------------------------------------------------------------
    # CAPTIONING MODEL
    #
    # In this example, we extract the features from the lower convolutional layer of
    # InceptionV3 giving us a vector of shape (8, 8, 2048). We squash that to a shape
    # of (64, 2048). This vector is then passed through the CNN Encoder(which consists of a
    # single Fully connected layer). The RNN(here GRU) attends over the image to predict
    # the next word.
    # -----------------------------------------------------------------------------------
    print("Building Captioning Model ...")

    dim_embedding = 512

    dim_model = 512
    dim_hidden = 512
    num_attention_heads = 8
    dim_key = 64
    dim_value = 64
    num_decoder_layers = 2
    prob_dropout = 0.1

    caption_model = CaptionTransformer(
        d_embed=dim_embedding,
        d_model=dim_model,
        d_hidden=dim_hidden,
        n_attn_heads=num_attention_heads,
        d_k=dim_key,
        d_v=dim_value,
        n_decoder_layers=num_decoder_layers,
        p_dropout=prob_dropout,
        max_capt_len=max_length,
        v_size=VOCAB_LENGTH,
    )

    caption_model.compile(optimizer=Adam(0.001, 0.9, 0.98, epsilon=1e-9))

    print("Saving Model Architecture")
    plot_model(caption_model.model, to_file= 'model.png', show_shapes=True)

    print("Training Model ...")
    start_time = datetime.now()

    history = caption_model.model.fit_generator(
        generator=train_data_generator,
        epochs=30,
        steps_per_epoch=468,
        verbose=1,
        validation_data=val_data_generator,
        validation_steps=10,
        # max_q_size=1,
        workers=8,
        # callbacks=training_cb
    )



    #
    # caption_model.model.fit(
    #     [Xtrain, Ytrain],
    #     None,
    #     batch_size=64,
    #     epochs=30,
    #     validation_data=([Xvalid, Yvalid], None),
    #     callbacks=[lr_scheduler, model_saver]
    # )

    print("Training took {}".format(datetime.now() - start_time))





    # # -----------------------------------------------------------------------------------
    # # Build Model
    # # -----------------------------------------------------------------------------------
    # max_caption_length = 70
    # dim_embedding = 512
    # vocab_size = 5001
    #
    # dim_model = 512
    # dim_hidden = 512
    # num_attention_heads = 8
    # dim_key = 64
    # dim_value = 64
    # num_decoder_layers = 2
    # prob_dropout = 0.1
    #
    # caption_model = CaptionTransformer(
    #     d_embed=dim_embedding,
    #     d_model=dim_model,
    #     d_hidden=dim_hidden,
    #     n_attn_heads=num_attention_heads,
    #     d_k=dim_key,
    #     d_v=dim_value,
    #     n_decoder_layers=num_decoder_layers,
    #     p_dropout=prob_dropout,
    #     max_capt_len=max_caption_length,
    #     v_size=vocab_size,
    # )
    #
    # caption_model.compile(optimizer=Adam(0.001, 0.9, 0.98, epsilon=1e-9))

    # -------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------


    # # -----------------------------------------------------------------------------------
    # # Get Data
    # # -----------------------------------------------------------------------------------
    # print("Getting Data ...")
    #
    # train_captions, img_name_vector = get_mscoco_data()
    # # -----------------------------------------------------------------------------------
    # # Image Encoder
    # # InceptionV3 model (pretrained on Imagenet). Feature Shape [2048, 64]
    # # -----------------------------------------------------------------------------------
    # print("Making image feature extracting network ...")
    #
    # # Don't include the top, we only need the features from the last layer, not the classifier
    # image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    # new_input = image_model.input
    # hidden_layer = image_model.layers[-1].output
    #
    # image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    #
    # # Store Features of Images and use these for training
    # # --------------------------------------------
    # print("Extracting Features from dataset images ...")
    #
    # # Get unique images, there are multiple captions per image. We only need to store
    # # features of unique images
    # IMAGE_CACHING_BATCH_SIZE = 16
    #
    # encode_train = sorted(set(img_name_vector))
    # image_dataset = tf.data.Dataset.from_tensor_slices(encode_train).map(load_image).batch(IMAGE_CACHING_BATCH_SIZE)
    #
    # start_feature_extract = datetime.now()
    # for img, path in tqdm(image_dataset):
    #
    #     batch_features = image_features_extract_model(img)
    #     batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
    #
    #     for bf, p in zip(batch_features, path):
    #         path_of_feature = p.numpy().decode("utf-8")
    #         np.save(path_of_feature, bf.numpy())
    #
    # print("Image Feature Extracting Step took {}".format(datetime.now() - start_feature_extract))
    #
    # # -----------------------------------------------------------------------------------
    # # Caption Preprocessing
    # # -----------------------------------------------------------------------------------
    # # 1. Tokenize the captions (e.g., by splitting on spaces) to generate Vocabulary
    # # 2. Limit Vocabulary to save memory, all other words replaced with token "UNK"
    # # 3. Create word -> index mapping (to easily translate between them)
    # # 4. Pad all captions to same (longest) length
    # print("Tokenizing Captions...")
    #
    # VOCAB_LENGTH = 5000
    #
    # tokenizer = tf.keras.preprocessing.text.Tokenizer(
    #     num_words=VOCAB_LENGTH,
    #     oov_token="<unk>",
    #     filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    #
    # tokenizer.fit_on_texts(train_captions)
    #
    # # Represent each word by its token index
    # # Eg. '<start> A skateboarder performing a trick on a skateboard ramp. <end>' ==>
    # # [3, 2, 351, 687, 2, 280, 5, 2, 84, 339, 4]
    # train_seqs = tokenizer.texts_to_sequences(train_captions)
    #
    # # Mapping word -> index
    # tokenizer.word_index = \
    #     {key: value for key, value in tokenizer.word_index.items() if value <= VOCAB_LENGTH}
    #
    # # Putting <unk> token in the word2idx dictionary
    # tokenizer.word_index[tokenizer.oov_token] = VOCAB_LENGTH + 1
    # tokenizer.word_index['<pad>'] = 0
    #
    # # # creating the tokenized vectors
    # # train_seqs = tokenizer.texts_to_sequences(train_captions)
    #
    # # Map index -> word
    # index_word = {value: key for key, value in tokenizer.word_index.items()}
    #
    # # padding each vector to the max_length of the captions
    # # array([  3,  29,  53,  19,  89, 202, 100,  92,   5,   7, 218,   4,   0,
    # #          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    # #          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    # #          0,   0,   0,   0,   0,   0,   0,   0,   0,   0], dtype=int32)
    # cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    #
    # # calculating the max_length
    # # used to store the attention weights
    # max_length = calc_max_length(train_seqs)
    #
    # # -----------------------------------------------------------------------------------
    # # Train Validation Split
    # # -----------------------------------------------------------------------------------
    # print("Creating Training and Validation Split ... ")
    #
    # TRAIN_VALIDATION_SPLIT = 0.2
    #
    # img_name_train, img_name_val, cap_train, cap_val = train_test_split(
    #     img_name_vector,
    #     cap_vector,
    #     test_size=TRAIN_VALIDATION_SPLIT,
    #     random_state=0)
    #
    # print("Training ({} images, {} captions). Validation ({} images, {} captions)".format(
    #     len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)))
    #
    # # -----------------------------------------------------------------------------------
    # # Create a Tensorflow DataSet
    # # -----------------------------------------------------------------------------------
    # print("Creating a Tensorflow Dataset ...")
    #
    # BATCH_SIZE = 64
    # BUFFER_SIZE = 1000
    # embedding_dim = 256
    #
    # units = 512
    # vocab_size = len(tokenizer.word_index)
    # # shape of the vector extracted from InceptionV3 is (64, 2048)
    # # these two variables represent that
    # features_shape = 2048
    # attention_features_shape = 64
    #
    # dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    #
    # # using map to load the numpy files in parallel
    # # NOTE: Be sure to set num_parallel_calls to the number of CPU cores you have
    # # https://www.tensorflow.org/api_docs/python/tf/py_func
    # dataset = dataset.map(lambda item1, item2: tf.py_func(
    #     map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=8)
    #
    # # shuffling and batching
    # dataset = dataset.shuffle(BUFFER_SIZE)
    # # https://www.tensorflow.org/api_docs/python/tf/contrib/data/batch_and_drop_remainder
    # dataset = dataset.batch(BATCH_SIZE)
    # dataset = dataset.prefetch(1)






