import tensorflow as tf
import numpy as np
import os
import time
from keras.models import load_model

# files= ['1SorcerersStone.txt', '2ChamberofSecrets.txt', '3ThePrisonerOfAzkaban.txt', '4TheGobletOfFire.txt', '5OrderofthePhoenix.txt', '6TheHalfBloodPrince.txt', '7DeathlyHollows.txt']
# with open('harrypotter.txt', 'w') as outfile:
# for file in files:
#  with open(file) as infile:
#   outfile.write(infile.read())

text = open('1SorcerersStone.txt').read()
print('Length of text: {} characters'.format(len(text)))

# Taking a look at the text
print(text[:300])

# The unique characters in the file
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters to indices
char2index = {u: i for i, u in enumerate(vocab)}
index2char = np.array(vocab)

text_as_int = np.array([char2index[c] for c in text])

print(text_as_int)

# Show how the first 13 characters from the text are mapped to integers
print('{} -- characters mapped to int -- > {}'.format(repr(text[:13]), text_as_int[:13]))

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# for i in char_dataset.take(5):
#  print(index2char[i.numpy()])

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


# for item in sequences.take(5):
#  print(repr(''.join(index2char[item.numpy()])))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 300  # 256

# Number of RNN units
rnn_units1 = 512  # 1024
rnn_units2 = 256
rnn_units = [rnn_units1, rnn_units2]
print(vocab_size)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units1,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.GRU(rnn_units2,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

model.summary()


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
EPOCHS = 1
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
latest_check = tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(latest_check)

model.build(tf.TensorShape([1, None]))

model.summary()

model.save('my_model.h5')
