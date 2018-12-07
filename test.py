# -*- coding: utf-8 -*-
# !/usr/bin/python3
# ---------------------------------------------------------------------------------------
# Image Captioning using the Transformer Network
#
# References:
# [1] https://github.com/Lsdefine/attention-is-all-you-need-keras
#     Transformer for machine translation.
#
# [2] https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/
#     python/examples/generative_examples/image_captioning_with_attention.ipynb
#    Image captioning using an RNN (GRU)
# ---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm import tqdm
import numpy as np
import glob
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as keras

from external.attention_is_all_you_need.transformer import get_pos_encoding_matrix, Decoder
from image_captioning_with_attention import CnnEncoder, get_mscoco_data, calc_max_length

BASE_RESULTS_DIR = 'models'


def load_image(img_path):
    """
    Preprocess Images for Inception V3 Model

    :param img_path:
    :return:
    """
    x = keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
    x = keras.preprocessing.image.img_to_array(x)
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    x = keras.applications.inception_v3.preprocess_input(x)  # image pixels range (-1, 1)

    return x


# noinspection PyAttributeOutsideInit
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_key_dict, b_size=32, feature_size=(64, 2048), max_cap_len=70, shuffle=True):
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
        self.max_caption_len = max_cap_len

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

            temp = np.load(list_id + '.npy')

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


# noinspection PyAttributeOutsideInit
class CaptionTransformer:
    def __init__(
            self, d_embed, d_model, d_hidden, n_attn_heads, d_k, d_v, n_decoder_layers,
            p_dropout, max_capt_len, v_size):

        self.encoder = CnnEncoder(d_embed)

        # Position encoding layer, this is used to preserve the order of words in caption sequences.
        self.pos_embed_layer = keras.layers.Embedding(
            max_capt_len,
            d_embed,
            trainable=False,
            weights=[get_pos_encoding_matrix(max_capt_len, d_embed)],
            name='position_embed'
        )

        # Word Embedding Layer
        self.word_embed_layer = keras.layers.Embedding(v_size, d_embed, name='word_embed')

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

        self.map_to_word_tokens_layer = \
            keras.layers.TimeDistributed(keras.layers.Dense(v_size, use_bias=False))

    @staticmethod
    def get_pos_seq(x):
        mask = keras.backend.cast(keras.backend.not_equal(x, 0), 'int32')
        pos = tf.cumsum(keras.backend.ones_like(x, 'int32'), 1)
        return pos * mask

    def compile(self, opt, active_layers=999):

        # This is for Teacher Forcing
        img_features = keras.layers.Input(shape=(64, 2048,), dtype='float32', name='image_features')
        tgt_seq_input = keras.layers.Input(shape=(None,), dtype='int32', name='true_caption')

        # Teacher Forcing caption
        tf_cap_seq = keras.layers.Lambda(lambda x1: x1[:, :-1], name='tf_cap_seq')(tgt_seq_input)
        true_cap_seq = keras.layers.Lambda(lambda x1: x1[:, 1:], name='true_cap_seq')(tgt_seq_input)

        # tf_cap_seq = [43, 10, 20, 45, 0, 0, 0, 0]  # tokenized words (sequence)
        # tgt_pos = [ 1,  2,  3,  4, 0, 0, 0, 0]
        tgt_pos = keras.layers.Lambda(self.get_pos_seq, name='map_tf_seq_to_pos_vector')(tf_cap_seq)

        enc_output = self.encoder(img_features)
        dec_output = self.decoder(tf_cap_seq, tgt_pos, None, enc_output, active_layers=active_layers)
        final_output = self.map_to_word_tokens_layer(dec_output)

        def get_loss(args):
            y_pred, y_true = args
            y_true = tf.cast(y_true, 'int32')
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss1 = tf.reduce_sum(loss1 * mask, -1) / tf.reduce_sum(mask, -1)
            loss1 = keras.backend.mean(loss1)
            return loss1

        def get_accuracy(args):
            y_pred, y_true = args
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = keras.backend.cast(
                keras.backend.equal(
                    keras.backend.cast(y_true, 'int32'),
                    keras.backend.cast(keras.backend.argmax(y_pred, axis=-1), 'int32')), 'float32')

            corr = keras.backend.sum(corr * mask, -1) / keras.backend.sum(mask, -1)

            return keras.backend.mean(corr)

        loss = keras.layers.Lambda(get_loss, name='get_loss')([final_output, true_cap_seq])

        self.ppl = keras.layers.Lambda(keras.backend.exp, name='exponential_loss')(loss)
        self.accu = keras.layers.Lambda(get_accuracy, name='get_accuracy')([final_output, true_cap_seq])

        self.training_model = keras.models.Model([img_features, tgt_seq_input], loss)
        self.training_model.add_loss([loss])

        self.training_model.compile(opt, None)
        self.training_model.metrics_names.append('ppl')
        self.training_model.metrics_tensors.append(self.ppl)
        self.training_model.metrics_names.append('accu')
        self.training_model.metrics_tensors.append(self.accu)

        # This Model does not do Teacher Forcing and should be used on test/validation data
        # ---------------------------------------------------------------------------------
        img_features_1 = keras.layers.Input(shape=(64, 2048,), dtype='float32', name='image_features_1')
        tgt_seq_input_1 = keras.layers.Input(shape=(None,), dtype='int32', name='decoder_output')
        tgt_seq_1 = tgt_seq_input_1

        tgt_pos = keras.layers.Lambda(self.get_pos_seq)(tgt_seq_1)

        enc_output_1 = self.encoder(img_features_1)
        dec_output_1 = self.decoder(tgt_seq_1, tgt_pos, None, enc_output_1,  active_layers=active_layers)
        final_output_1 = self.map_to_word_tokens_layer(dec_output_1)

        self.prediction_model = keras.models.Model([img_features_1, tgt_seq_input_1], final_output_1)
        self.prediction_model.compile(opt, 'mean_squared_error')


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    results_identifier = 'captioning_transformer'

    # Immutable
    if not os.path.exists(BASE_RESULTS_DIR):
        os.mkdir(BASE_RESULTS_DIR)

    results_dir = os.path.join(BASE_RESULTS_DIR, results_identifier)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # -----------------------------------------------------------------------------------
    # Get Data
    # -----------------------------------------------------------------------------------
    print("Getting Data {}".format('.' * 80))
    data_captions, data_img_names = get_mscoco_data(n_train=30000)

    # -----------------------------------------------------------------------------------
    # Image Feature Encoding Model
    # -----------------------------------------------------------------------------------
    # Don't include the top, we only need the features from the last layer, not the classifier
    image_model = keras.applications.InceptionV3(include_top=False, weights='imagenet')

    image_features_extract_model = tf.keras.Model(
        image_model.input,
        image_model.layers[-1].output)

    # -----------------------------------------------------------------------------------
    # Image Feature Extraction & Storing
    # -----------------------------------------------------------------------------------
    print("Extracting image features and storing {}".format('.' * 80))

    start_feature_extract = datetime.now()

    # Get unique images, there are multiple captions per image. We only need to store
    # features of unique images
    unique_image_names = set(data_img_names)

    par_dir = os.path.dirname(data_img_names[0])
    np_files = glob.glob(par_dir + '/*.npy')
    if len(np_files) == len(unique_image_names):
        print("Image features already extracted")

    else:
        for idx, image_path in tqdm(enumerate(unique_image_names)):
            preprocessed_img = load_image(image_path)
            img_feat = image_features_extract_model.predict(preprocessed_img, verbose=0)
            img_feat = np.reshape(img_feat, (img_feat.shape[0], -1, img_feat.shape[3]))
            img_features_path = image_path + '.npy'
            np.save(img_features_path, img_feat)

        print("Image feature extracting step took {}".format(datetime.now() - start_feature_extract))

    # -----------------------------------------------------------------------------------
    # Caption Preprocessing
    # -----------------------------------------------------------------------------------
    # 1. Tokenize the captions (e.g., by splitting on spaces) to generate Vocabulary
    # 2. Limit Vocabulary to save memory, all other words replaced with token "UNK"
    # 3. Create word -> index mapping (to easily translate between them)
    # 4. Pad all captions to same (longest) length
    print("Tokenizing Captions {}".format('.' * 80))

    VOCAB_LENGTH = 5000

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=VOCAB_LENGTH,
        oov_token="<unk>",
        filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(data_captions)

    # Represent each word by its token index
    # Eg. '<start> A skateboarder performing a trick on a skateboard ramp. <end>' ==>
    # [3, 2, 351, 687, 2, 280, 5, 2, 84, 339, 4]
    train_seqs = tokenizer.texts_to_sequences(data_captions)

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

    max_caption_len = calc_max_length(train_seqs)

    # -----------------------------------------------------------------------------------
    # Train Validation Split
    # -----------------------------------------------------------------------------------
    print("Creating Training/Validation data splits {}".format('.' * 80))

    TRAIN_VALIDATION_SPLIT = 0.2

    train_image_names, val_image_names, train_cap, val_cap = train_test_split(
        data_img_names,
        cap_vector,
        test_size=TRAIN_VALIDATION_SPLIT,
        random_state=0)

    print("Training ({} images, {} captions). Validation ({} images, {} captions)".format(
        len(train_image_names), len(train_cap), len(val_image_names), len(val_cap)))

    num_train = len(train_image_names)
    num_val = len(val_image_names)

    print("Training Data: N = {}. Num unique images {}".format(num_train, len(set(train_image_names))))
    print("Validation Data: N = {}. Num unique images {}".format(num_val, len(set(val_image_names))))

    # -----------------------------------------------------------------------------------
    # Create Data  Generators
    # -----------------------------------------------------------------------------------
    print("Creating Data Generators {}".format('.' * 80))
    extracted_img_feature_dim = (64, 2048)
    batch_size = 64

    train_data_dict = {}
    for idx in range(len(train_image_names)):
        train_data_dict[train_image_names[idx]] = train_cap[idx]

    train_data_generator = DataGenerator(
        train_data_dict,
        b_size=batch_size,
        shuffle=True,
        max_cap_len=max_caption_len,
        feature_size=extracted_img_feature_dim
    )

    # gen_out = iter(train_data_generator)
    # train_images, train_labels = gen_out.__next__()

    valid_data_dict = {}
    for idx in range(len(val_image_names)):
        valid_data_dict[val_image_names[idx]] = val_cap[idx]

    val_data_generator = DataGenerator(
        valid_data_dict,
        b_size=batch_size,
        shuffle=True,
        max_cap_len=max_caption_len,
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
    print("Building Captioning Model {}".format('.' * 80))

    dim_embedding = 512

    dim_model = 512
    dim_hidden = 512
    num_attention_heads = 8
    dim_key = 64
    dim_value = 64
    num_decoder_layers = 6
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
        max_capt_len=max_caption_len,
        v_size=VOCAB_LENGTH,
    )

    optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = keras.optimizers.Adam(0.0001, 0.9, 0.98, epsilon=1e-9)
    caption_model.compile(optimizer)

    print("Saving Model Architecture ")
    keras.utils.plot_model(
        caption_model.training_model,
        to_file=os.path.join(results_dir, 'model.png'),
        show_shapes=True)

    caption_model.training_model.summary()

    # -----------------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------------
    print("Training {}".format('.' * 80))

    num_epochs = 100
    start_time = datetime.now()


    def learning_rate_modifier(epoch_idx, curr_learning_rate):
        if epoch_idx == (num_epochs // 2.0):
            curr_learning_rate = curr_learning_rate / 10.0
        elif epoch_idx == (num_epochs // 4.0 * 3):
            curr_learning_rate = curr_learning_rate / 10.0

        return curr_learning_rate

    learning_rate_modifying_cb = keras.callbacks.LearningRateScheduler(
        learning_rate_modifier,
        verbose=1
    )

    model_save_file = os.path.join(results_dir, 'weights.h5')
    model_saver = keras.callbacks.ModelCheckpoint(model_save_file, save_best_only=True, save_weights_only=True)

    history = caption_model.training_model.fit_generator(
        generator=train_data_generator,
        epochs=num_epochs,
        steps_per_epoch=(num_train // batch_size),
        verbose=1,
        validation_data=val_data_generator,
        validation_steps=(num_val // batch_size),
        # max_q_size=1,
        workers=8,
        callbacks=[model_saver, learning_rate_modifying_cb]
    )
    print("Training took {}".format(datetime.now() - start_time))

    f, ax_arr = plt.subplots(1, 2)
    ax_arr[0].plot(history.history['loss'], label='train', color='b')
    ax_arr[0].plot(history.history['val_loss'], label='validation', color='r')
    ax_arr[0].set_xlabel("Epochs")
    ax_arr[0].set_ylabel("Loss")

    ax_arr[1].plot(history.history['accu'], label='train', color='b')
    ax_arr[1].plot(history.history['val_accu'], label='validation', color='r')
    ax_arr[1].set_xlabel("Epochs")
    ax_arr[1].set_ylabel("Accuracy")
    ax_arr[1].legend()

    f.savefig(os.path.join(results_dir, 'training.eps'), format='eps')

    summary_file = os.path.join(results_dir, 'summary.text')
    with open(summary_file, 'w') as handle:
        handle.write("Final Train Loss: {}\n".format(history.history['loss'][-1]))
        handle.write("Final Validation Loss: {}\n".format(history.history['val_loss'][-1]))
        handle.write("Final Train Accuracy: {}\n".format(history.history['accu'][-1]))
        handle.write("Final Validation Accuracy: {}\n".format(history.history['val_accu'][-1]))
        handle.write("\n")
        handle.write("Number of parameters {}\n".format(caption_model.training_model.count_params()))
        handle.write("Number of attention heads {}\n".format(num_attention_heads))
        handle.write("Number of Decoder Layers heads {}\n".format(num_decoder_layers))
        handle.write("\n")
        handle.write("Number of Epochs {}\n".format(num_epochs))

    # -----------------------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------------------
    print("Sample predictions {}".format('.' * 80))

    # Sample image
    sample_img_idx_arr = [201, 10, 5, 500, 30]

    for sample_img_idx in sample_img_idx_arr:
        example_img_name = data_img_names[sample_img_idx]
        # TODO: Start from images that dont have a start and end
        example_img_caption = data_captions[sample_img_idx]

        from PIL import Image
        temp_image = np.array(Image.open(example_img_name))
        plt.figure()
        plt.imshow(temp_image)

        # Extract hidden layer features
        x_img = load_image(example_img_name)
        x_img_features = image_features_extract_model(keras.backend.expand_dims(x_img[0], axis=0))
        hidden_feature_input = tf.reshape(x_img_features, (x_img_features.shape[0], -1, x_img_features.shape[3]))

        # Tokenize the caption:
        # Expects a list of captions
        x_img_cap = tokenizer.texts_to_sequences([example_img_caption])
        # x_img_cap = tf.keras.preprocessing.sequence.pad_sequences(x_img_cap, maxlen=max_caption_len, padding='post')

        decoded_tokens = []
        target_seq = np.zeros((1, max_caption_len), dtype='int32')
        target_seq[0, 0] = tokenizer.word_index['<start>']

        for i in range(max_caption_len - 1):
            output = caption_model.prediction_model.predict_on_batch([hidden_feature_input, target_seq])
            sampled_index = np.argmax(output[0, i, :])
            sampled_token = tokenizer.index_word[sampled_index]
            print("{}: output word: {}".format(i, sampled_token))
            decoded_tokens.append(sampled_token)

            if sampled_token == '<end>':
                break

            target_seq[0, i + 1] = sampled_index

        print("Decoded: {}".format(' '.join(decoded_tokens[:-1])))

        example_img_caption = example_img_caption.replace('<start>', '')
        example_img_caption = example_img_caption.replace('<end>', '')
        print("True: {}".format(example_img_caption))

        plt.title("True: {}\n Predicted: {} ".format(example_img_caption, ' '.join(decoded_tokens[:-1])), loc='left')
        f = plt.gcf()
        f.savefig(os.path.join(results_dir, 'sample_caption_{}.eps'.format(sample_img_idx)), format='eps')
