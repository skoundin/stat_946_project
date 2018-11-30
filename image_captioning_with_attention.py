# -*- coding: utf-8 -*-
# !/usr/bin/python3
# ---------------------------------------------------------------------------------------
# Show Attend and Tell
#
# REF: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/
# eager/python/examples/generative_examples/image_captioning_with_attention.ipynb
# ---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import json
from PIL import Image
from tqdm import tqdm

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# ***************************************************************************************
#  Get the Data
# ***************************************************************************************
def get_mscoco_data(n_train=30000):
    """
    Get MS COCO Data

    :return: 2 lists: [0] list of captions, [1] list of image corresponding to the caption
    """
    print("Getting MS COCO DataSet ...")

    data_dir = os.path.join(os.path.abspath('.'), 'data/mscoco')

    print("Getting Captions Files...")
    annotation_zip = tf.keras.utils.get_file(
        'captions.zip',
        cache_subdir=data_dir,
        origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        extract=True)
    annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'

    print("Getting Training Files...")
    name_of_zip = 'train2014.zip'
    if not os.path.exists(data_dir + '/' + name_of_zip):
        image_zip = tf.keras.utils.get_file(
            name_of_zip,
            cache_subdir=data_dir,
            origin='http://images.cocodataset.org/zips/train2014.zip',
            extract=True)

        img_path = os.path.dirname(image_zip) + '/train2014/'
    else:
        img_path = data_dir + '/train2014/'

    # Pre-processing
    # --------------
    print("Format data for ease of use...")

    # Read the annotations file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # storing the captions and the image name in vectors
    all_captions = []
    all_img_name_vector = []

    for annotation in annotations['annotations']:
        caption = '<start> ' + annotation['caption'] + ' <end>'
        image_id = annotation['image_id']
        full_coco_image_path = img_path + 'COCO_train2014_' + '%012d.jpg' % image_id

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # shuffling the captions and image_names together
    # setting a random state
    t_captions, img_name_arr = shuffle(all_captions, all_img_name_vector, random_state=1)

    # Limit the training set for faster Training

    t_captions = t_captions[:n_train]
    img_name_arr = img_name_arr[:n_train]

    print("Training with {} captions. Total Number of captions in dataset {}".format(
        len(t_captions), len(all_captions)))

    return t_captions, img_name_arr


def load_image(img_path):
    """
    Preprocess Images for Inception V3 Model

    :param img_path:
    :return:
    """
    x = tf.read_file(img_path)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize_images(x, (299, 299))
    x = tf.keras.applications.inception_v3.preprocess_input(x)  # image pixels range (-1, 1)

    return x, img_path


def calc_max_length(tensor):
    """
    Find the maximum length of any caption in our dataset

    :param tensor:
    :return:
    """
    return max(len(t) for t in tensor)


def map_func(img_name, cap):
    """
    loading the numpy files

    :param img_name:
    :param cap:
    :return:
    """
    x = np.load(img_name.decode('utf-8')+'.npy')
    return x, cap


# ***************************************************************************************
# DECODER FUNCTIONS
# ***************************************************************************************
def gru(num_units):
    """

    :param num_units:
    :return:
    """
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(
            num_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')

    else:
        return tf.keras.layers.GRU(
            num_units,
            return_sequences=True,
            return_state=True,
            recurrent_activation='sigmoid',
            recurrent_initializer='glorot_uniform')


class BahdanauAttention(tf.keras.Model):
    def __init__(self, num_units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(num_units)
        self.W2 = tf.keras.layers.Dense(num_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CnnEncoder(tf.keras.Model):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embed_dim):
        super(CnnEncoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embed_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RnnDecoder(tf.keras.Model):
    def __init__(self, embed_dim, num_units, v_size):
        super(RnnDecoder, self).__init__()
        self.units = num_units

        self.embedding = tf.keras.layers.Embedding(v_size, embed_dim)
        self.gru = gru(self.units)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(v_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


# ***************************************************************************************
# LOSS FUNCTIONS
# ***************************************************************************************
# We are masking the loss calculated for padding
def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


def evaluate(x):
    attention_plt = np.zeros((max_length, attention_features_shape))

    hidden1 = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(x)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features1 = encoder(img_tensor_val)

    dec_input1 = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result1 = []

    for ii in range(max_length):
        print("Iteration {} of {}".format(ii, max_length))
        predictions1, hidden1, attention_weights = decoder(dec_input1, features1, hidden1)

        attention_plt[ii] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions1[0]).numpy()
        result1.append(index_word[predicted_id])

        if index_word[predicted_id] == '<end>':
            return result1, attention_plt

        dec_input1 = tf.expand_dims([predicted_id], 0)

    attention_plt = attention_plt[:len(result1), :]
    return result1, attention_plt


def plot_attention(x, result1, attention_plt):
    temp_image = np.array(Image.open(x))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result1)
    for l in range(len_result):
        temp_att = np.resize(attention_plt[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result1[l])
        img1 = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img1.get_extent())

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    tf.enable_eager_execution()

    # -----------------------------------------------------------------------------------
    # Download data &  Make available for efficient use
    # MS-COCO:  This dataset contains >82,000 images, each of which has been annotated
    # with at least 5 different captions.
    # -----------------------------------------------------------------------------------
    print("Getting Data ...")
    train_captions, img_name_vector = get_mscoco_data()

    # -----------------------------------------------------------------------------------
    # Image Encoder
    # InceptionV3 model (pretrained on Imagenet). Feature Shape [2048, 64]
    # -----------------------------------------------------------------------------------
    print("Making image feature extracting network ...")

    # Don't include the top, we only need the features from the last layer, not the classifier
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # --------------------------------------------
    # Image Preprocessing
    # --------------------------------------------
    # Instead of processing the images, extract features from each image and 'cache' them, for computational speed
    print("Caching Features of the Images ...")

    # Getting the unique images
    IMAGE_CACHING_BATCH_SIZE = 16

    # TODO(Salman): Why is is necessary ?
    encode_train = sorted(set(img_name_vector))

    # feel free to change the batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train).map(load_image).batch(IMAGE_CACHING_BATCH_SIZE)

    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

    # --------------------------------------------
    # Caption Preprocessing
    # --------------------------------------------
    # 1. Tokenize the captions (e.g., by splitting on spaces) to generate Vocabulary
    # 2. Limit Vocabulary to save memory, all other words replaced with token "UNK"
    # 3. Create word -> index mapping (to easily translate between them)
    # 4. Pad all captions to same (longest) length
    print("Preprocessing Captions")

    VOCAB_LENGTH = 5000

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=VOCAB_LENGTH,
        oov_token="<unk>",
        filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Mapping word -> index
    tokenizer.word_index = {key: value for key, value in tokenizer.word_index.items() if value <= VOCAB_LENGTH}

    # putting <unk> token in the word2idx dictionary
    tokenizer.word_index[tokenizer.oov_token] = VOCAB_LENGTH + 1
    tokenizer.word_index['<pad>'] = 0

    # # creating the tokenized vectors
    # train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Map index -> word
    index_word = {value: key for key, value in tokenizer.word_index.items()}

    # padding each vector to the max_length of the captions
    # if the max_length parameter is not provided, pad_sequences calculates that automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # calculating the max_length
    # used to store the attention weights
    max_length = calc_max_length(train_seqs)

    # -----------------------------------------------------------------------------------
    # Train Validation Split
    # -----------------------------------------------------------------------------------
    print("Creating Training and Validation Split")

    TRAIN_VALIDATION_SPLIT = 0.2

    img_name_train, img_name_val, cap_train, cap_val = train_test_split(
        img_name_vector,
        cap_vector,
        test_size=TRAIN_VALIDATION_SPLIT,
        random_state=0)

    print("Training ({} images, {} captions). Validation ({} images, {} captions)".format(
        len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)))

    """## Our images and captions are ready! Next, let's create a tf.data dataset to use for training our model."""
    # -----------------------------------------------------------------------------------
    # Create a Tensorflow DataSet
    # -----------------------------------------------------------------------------------
    print("Creating a tensorflow Dataset ...")

    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    embedding_dim = 256

    units = 512
    vocab_size = len(tokenizer.word_index)
    # shape of the vector extracted from InceptionV3 is (64, 2048)
    # these two variables represent that
    features_shape = 2048
    attention_features_shape = 64

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # using map to load the numpy files in parallel
    # NOTE: Be sure to set num_parallel_calls to the number of CPU cores you have
    # https://www.tensorflow.org/api_docs/python/tf/py_func
    dataset = dataset.map(lambda item1, item2: tf.py_func(
          map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=8)

    # shuffling and batching
    dataset = dataset.shuffle(BUFFER_SIZE)
    # https://www.tensorflow.org/api_docs/python/tf/contrib/data/batch_and_drop_remainder
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    # -----------------------------------------------------------------------------------
    # DECODER:
    #
    # In this example, we extract the features from the lower convolutional layer of
    # InceptionV3 giving us a vector of shape (8, 8, 2048). We squash that to a shape
    # of (64, 2048). This vector is then passed through the CNN Encoder(which consists of a
    # single Fully connected layer). The RNN(here GRU) attends over the image to predict
    # the next word.
    # -----------------------------------------------------------------------------------
    print("Building Decoder ...")

    encoder = CnnEncoder(embedding_dim)
    decoder = RnnDecoder(embedding_dim, units, vocab_size)

    # -----------------------------------------------------------------------------------
    # Training
    # 1. Extract image features stored in '.npy' files and pass through CNN encoder (Not image encoder)
    # 2. The encoder output, hidden state (initialized to 0) and the decoder input
    #    (which is the start token) is passed to the decoder.
    # 3. The decoder returns the predictions and the decoder hidden state.
    # 4. The decoder hidden state is then passed back into the model and the predictions
    #    are used to calculate the loss.
    # 5. Use teacher forcing to decide the next input to the decoder. Teacher forcing is the technique
    #    where the target word is passed as the next input to the decoder.
    # 6. The final step is to calculate the gradients and apply it to the optimizer and backpropagate.
    # -----------------------------------------------------------------------------------
    print("Start Training ...")
    optimizer = tf.train.AdamOptimizer()

    # adding this in a separate cell because if you run the training cell
    # many times, the loss_plot array will be reset
    loss_plot = []

    EPOCHS = 20

    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            loss = 0

            # initializing the hidden state for each batch
            # because the captions are not related from image to image
            hidden = decoder.reset_state(batch_size=target.shape[0])

            dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

            with tf.GradientTape() as tape:
                features = encoder(img_tensor)

                for i in range(1, target.shape[1]):
                    # passing the features through the decoder
                    predictions, hidden, _ = decoder(dec_input, features, hidden)

                    loss += loss_function(target[:, i], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(target[:, i], 1)

            total_loss += (loss / int(target.shape[1]))

            variables = encoder.variables + decoder.variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1,
                    batch,
                    loss.numpy() / int(target.shape[1])))

        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / len(cap_vector))

        print('Epoch {} Loss {:.6f}'.format(
            epoch + 1,
            total_loss / len(cap_vector)))

        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # PLot loss function
    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')

    # -----------------------------------------------------------------------------------
    # Evaluate
    # The evaluate function is similar to the training loop, except we don't use teacher forcing here.
    # The input to the decoder at each time step is its previous predictions along with the hidden
    # state and the encoder output.
    # Stop predicting when the model predicts the end token.
    #  And store the attention weights for every time step.
    # -----------------------------------------------------------------------------------
    print("Starting Evaluation ...")

    # captions on the validation set
    rid = np.random.randint(0, len(img_name_val))
    image = img_name_val[rid]
    real_caption = ' '.join([index_word[i] for i in cap_val[rid] if i not in [0]])
    result, attention_plot = evaluate(image)

    print('Real Caption:', real_caption)
    print('Prediction Caption:', ' '.join(result))
    plot_attention(image, result, attention_plot)

    # opening the image
    Image.open(img_name_val[rid])

    """## Try it on your own images
    For fun, below we've provided a method you can use to caption your own images with the model we've just trained. 
     Keep in mind, it was trained on a relatively small amount of data, and your images may be different from the t
     raining data (so be prepared for weird results!)
    """

    image_url = 'https://tensorflow.org/images/surf.jpg'
    image_extension = image_url[-4:]
    image_path = tf.keras.utils.get_file('image'+image_extension, origin=image_url)

    result, attention_plot = evaluate(image_path)
    print('Prediction Caption:', ' '.join(result))
    plot_attention(image_path, result, attention_plot)
    # opening the image
    Image.open(image_path)
