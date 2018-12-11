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
import json
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import glob
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

BASE_RESULTS_DIR = 'results'
# ***************************************************************************************
#  Get the Data
# ***************************************************************************************
def get_mscoco_data(n_train=30000):
    """
    Get MS COCO Data.
    MS-COCO:  This dataset contains >82,000 images, each of which has been annotated
    with at least 5 different captions.

    If there are multiple captions per image. The image should be repeated in the
    list with the different caption as a separate entry.

    :return: 2 lists: [0] list of captions, [1] location of image file for the captions.
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

    # Generate lists of images and captions
    # -------------------------------------
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


# noinspection PyMethodOverriding
class BahdanauAttention(tf.keras.Model):
    def __init__(self, num_units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(num_units)
        self.W2 = tf.keras.layers.Dense(num_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features_in, hidden_in):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden_in, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features_in) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features_in
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# noinspection PyMethodOverriding
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


# noinspection PyMethodOverriding
class RnnDecoder(tf.keras.Model):
    def __init__(self, embed_dim, num_units, v_size):
        super(RnnDecoder, self).__init__()
        self.units = num_units

        self.embedding = tf.keras.layers.Embedding(v_size, embed_dim)
        self.gru = gru(self.units)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(v_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, dec_in, features_in, prev_hidden):
        """
        
        :param dec_in: Full word (not embedded). Either Start or previous output of decoder 
        :param features_in: Encoded Image Features
        :param prev_hidden: previous hidden state of the Decoder
        :return: 
        """
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features_in, prev_hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        dec_in = self.embedding(dec_in)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        dec_in = tf.concat([tf.expand_dims(context_vector, 1), dec_in], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(dec_in)

        # shape == (batch_size, max_length, hidden_size)
        dec_in = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        dec_in = tf.reshape(dec_in, (-1, dec_in.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        dec_in = self.fc2(dec_in)

        return dec_in, state, attention_weights

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
        # print("Iteration {} of {}".format(ii, max_length))
        predictions1, hidden1, attention_weights = decoder(dec_input1, features1, hidden1)

        attention_plt[ii] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions1[0]).numpy()
        if predicted_id == 1:
            predicted_id = 0
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
    for lx in range(len_result):
        temp_att = np.resize(attention_plt[lx], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, lx + 1)
        ax.set_title(result1[lx])
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

    results_identifier = 'gru_image_captioning'
    if not os.path.exists(BASE_RESULTS_DIR):
        os.mkdir(BASE_RESULTS_DIR)

    results_dir = os.path.join(BASE_RESULTS_DIR, results_identifier)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)


    # -----------------------------------------------------------------------------------
    # Get Data
    # -----------------------------------------------------------------------------------
    print("Getting Data ...")

    from flickr8k import get_flickr8k_data
    # train_captions, img_name_vector = get_mscoco_data()
    train_captions, img_name_vector = get_flickr8k_data()
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

    # Store Features of Images and use these for training
    # --------------------------------------------
    print("Extracting Features from dataset images ...")

    # Get unique images, there are multiple captions per image. We only need to store
    # features of unique images
    IMAGE_CACHING_BATCH_SIZE = 16

    encode_train = sorted(set(img_name_vector))
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train).map(load_image).batch(IMAGE_CACHING_BATCH_SIZE)

    start_feature_extract = datetime.now()
    unique_image_names = set(img_name_vector)
    par_dir = os.path.dirname(img_name_vector[0])
    np_files = glob.glob(par_dir + '/*.npy')

    if len(np_files) == len(unique_image_names):
        print("Image features already extracted")
    else:
        for img, path in tqdm(image_dataset, total=len(encode_train)):

            batch_features = image_features_extract_model(img)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())

    print("Image Feature Extracting Step took {}".format(datetime.now() - start_feature_extract))

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
        random_state=0,
        shuffle=False)

    print("Training ({} images, {} captions). Validation ({} images, {} captions)".format(
        len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)))

    # -----------------------------------------------------------------------------------
    # Create a Tensorflow DataSet
    # -----------------------------------------------------------------------------------
    print("Creating a Tensorflow Dataset ...")

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

    dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))

    # using map to load the numpy files in parallel
    # NOTE: Be sure to set num_parallel_calls to the number of CPU cores you have
    # https://www.tensorflow.org/api_docs/python/tf/py_func
    dataset_val = dataset_val.map(lambda item1, item2: tf.py_func(
          map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=8)

    # shuffling and batching
    dataset_val = dataset_val.shuffle(BUFFER_SIZE)
    # https://www.tensorflow.org/api_docs/python/tf/contrib/data/batch_and_drop_remainder
    dataset_val = dataset_val.batch(BATCH_SIZE)
    dataset_val = dataset_val.prefetch(1)

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
    
    loss_plot_val = []

    EPOCHS = 20
    
    for epoch in range(EPOCHS):
        start = datetime.now()
        total_loss = 0
        total_loss_val = 0
        count = 0
        count_val = 0
        for (batch, (img_tensor, true_caption)) in enumerate(dataset):
            loss = 0
            if true_caption.shape[0] % 64 == 0:
                # initializing the hidden state for each batch
                # because the captions are not related from image to image
                hidden = decoder.reset_state(batch_size=true_caption.shape[0])

                dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

                with tf.GradientTape() as tape:
                    features = encoder(img_tensor)

                    for i in range(1, true_caption.shape[1]):
                        # passing the features through the decoder
                        predictions, hidden, _ = decoder(dec_input, features, hidden)

                        loss += loss_function(true_caption[:, i], predictions)

                        # using teacher forcing
                        dec_input = tf.expand_dims(true_caption[:, i], 1)
                count = count + 1
                total_loss += (loss / int(true_caption.shape[1]))

                variables = encoder.variables + decoder.variables

                gradients = tape.gradient(loss, variables)

                optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1,
                        batch,
                        loss.numpy() / int(true_caption.shape[1])))

        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / count)
        total_loss = total_loss / count
        print('Epoch {} of {} Loss {:.6f}'.format(
            epoch + 1,
            EPOCHS,
            total_loss ))


        print('Getting Validation loss')
        for (batch, (img_tensor, true_caption)) in enumerate(dataset_val):
            loss_val = 0
            if true_caption.shape[0] % 64 == 0:
                # initializing the hidden state for each batch
                # because the captions are not related from image to image
                hidden = decoder.reset_state(batch_size=true_caption.shape[0])

                dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

                with tf.GradientTape() as tape:
                    features = encoder(img_tensor)

                    for i in range(1, true_caption.shape[1]):
                        # passing the features through the decoder
                        predictions, hidden, _ = decoder(dec_input, features, hidden)

                        loss_val += loss_function(true_caption[:, i], predictions)

                        # using teacher forcing
                        # dec_input = tf.expand_dims(true_caption[:, i], 1)
                count_val = count_val + 1
                total_loss_val += (loss_val / int(true_caption.shape[1]))

                #variables = encoder.variables + decoder.variables

                #gradients = tape.gradient(loss, variables)

                #optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())

                if batch % 100 == 0:
                    print('Epoch {} Batch {} validation Loss {:.4f}'.format(
                        epoch + 1,
                        batch,
                        loss_val.numpy() / int(true_caption.shape[1])))

        print('Validation loss for epoch {} is {}'.format(epoch,
            total_loss_val/count_val))
        print('Time taken for epoch {} sec\n'.format(datetime.now() - start))
        
        total_loss_val = total_loss_val / count_val
        loss_plot_val.append(total_loss_val)
        

    # PLot loss function
    plt.plot(loss_plot_val)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    
    plt.savefig(os.path.join(results_dir, 'training.eps'), format='eps')
    
    val_loss_arr = []
    for val_arr_i in  range(len(loss_plot_val)):
      l1 = loss_plot_val[0].numpy()
      val_loss_arr.append(l1)

    val_loss_file = os.path.join(results_dir, 'val_loss.pkl')  
    with open(val_loss_file, 'wb') as handle:
          pickle.dump(val_loss_arr, handle)
        
       

    # -----------------------------------------------------------------------------------
    # Evaluate
    # The evaluate function is similar to the training loop, except we don't use teacher forcing here.
    # The input to the decoder at each time step is its previous predictions along with the hidden
    # state and the encoder output.
    # Stop predicting when the model predicts the end token.
    #  And store the attention weights for every time step.
    # -----------------------------------------------------------------------------------
    # print("Starting Evaluation ...")
    # # captions on the validation set
    # for x in xrange(5):
    #     rid = np.random.randint(0, len(img_name_val))
    #     image = img_name_val[rid]
    #     real_caption = ' '.join([index_word[i] for i in cap_val[rid] if i not in [0]])
    #     result, attention_plot = evaluate(image)

    #     print('Real Caption:', real_caption)
    #     print('Prediction Caption:', ' '.join(result))
    #     plot_attention(image, result, attention_plot)

    #     # opening the image
    #     # Image.open(img_name_val[rid])

    #     plt.title("True: {}\n Predicted: {} ".format(real_caption, ' '.join(result)), loc='left')
    #     f = plt.gcf()
    #     f.savefig(os.path.join(results_dir, 'sample_caption_{}.eps'.format(sample_img_idx)), format='eps')

    # -------------------------------------------------------
    #                  Sample Captions
    # -------------------------------------------------------
    for x in range(5):
        rid = np.random.randint(0, len(img_name_val))
        image = img_name_val[rid]
        real_caption = ' '.join([index_word[i] for i in cap_val[rid] if i not in [0,1]])
        result, attention_plot = evaluate(image)

        print('Real Caption:', real_caption)
        print('Prediction Caption:', ' '.join(result))
        #plot_attention(image, result, attention_plot)
        from PIL import Image

        temp_image = np.array(Image.open(image))
        plt.figure()
        plt.imshow(temp_image)
        # opening the image
        # Image.open(img_name_val[rid])

        plt.title("True: {}\n Predicted: {} ".format(real_caption, ' '.join(result)), loc='left')
        f = plt.gcf()
        f.savefig(os.path.join(results_dir, 'sample_caption_{}.eps'.format(x)), format='eps')
    
    # -------------------------------------------------------
    #                  Get BLEU Scores
    # -------------------------------------------------------
    print("Calculating BLEU ... ")
    start_time = datetime.now()

    smoothie = SmoothingFunction()
    actual, predicted = list(), list()
    unique_val_img = set(img_name_val)

    for img in tqdm(unique_val_img, total=(len(unique_val_img))):

        indices = [i for i, x in enumerate(img_name_val) if x == img]
        real_caption = []
        for j in indices:
            real_caption.append(' '.join([index_word[i] for i in cap_val[j] if i not in [0, 1]]))
        ref = []
        for d in real_caption:
            l_temp = d.split()
            l_temp.remove('<start>')
            l_temp.remove('<end>')
            ref.append(l_temp)
        actual.append(ref)
        result, attention_plot = evaluate(img)
        if result[-1] == '<end>':
            result.remove('<end>')
        predicted.append(result)
    
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smoothie.method4)
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie.method4)
    bleu3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0), smoothing_function=smoothie.method4)
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie.method4)
    print("BLEU-1: {}".format(bleu1))
    print("BLEU-2: {}".format(bleu2))
    print("BLEU-3: {}".format(bleu3))
    print("BLEU-4: {}".format(bleu4))

    print("Calculating Blue score took: {}".format(datetime.now() - start_time))
    
    print('encoder variables')
    count_enc = 0
    for i in range(len(encoder.variables)):
      # print(encoder.variables[i].shape)
      param_v = 1
      for x in range(len(encoder.variables[i].shape)):
        param_v = param_v * encoder.variables[i].shape[x]
      count_enc = count_enc + param_v
    print(count_enc)

    print('decoder variables')
    count_dec = 0
    for i in range(len(decoder.variables)):
      param_v = 1
      # print(decoder.variables[i].shape)
      for x in range(len(decoder.variables[i].shape)):
        param_v = param_v * decoder.variables[i].shape[x]
      count_dec = count_dec + param_v
    print(count_dec)  
    print('total variables',count_enc + count_dec)
    
    summary_file = os.path.join(results_dir, 'summary.text')
    with open(summary_file, 'w') as handle:
        handle.write("Final Train Loss: {}\n".format(total_loss))
        handle.write("Final Validation Loss: {}\n".format(total_loss_val))
        handle.write("\n")
        handle.write("Number of parameters {}\n".format(count_enc + count_dec))
        handle.write("\n")
        handle.write("Number of Epochs {}\n".format(EPOCHS))
        handle.write("\n")
        handle.write("BLEU-1 {}\n".format(bleu1))
        handle.write("BLEU-2 {}\n".format(bleu2))
        handle.write("BLEU-3 {}\n".format(bleu3))
        handle.write("BLEU-4 {}\n".format(bleu4))
