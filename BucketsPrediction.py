import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

stop = list(stopwords.words('english'))

df_1 = pd.read_csv("C:/Users/palak/Desktop/ShopHopper/Data/dataLatest/p.csv",  error_bad_lines=False)
df = df_1.dropna()

print("Length of dataframe", len(df))

total_duplicate_titles = sum(df["titles"].duplicated())
print(f"There are {total_duplicate_titles} duplicate titles.")

df = df[~df["titles"].duplicated()]
print(f"There are {len(df)} rows in the deduplicated dataset.")

# There are some terms with occurrence as low as 1.
print(sum(df["buckets"].value_counts() == 1))

# How many unique terms?
print(df["buckets"].nunique())

arxiv_data_filtered = df.groupby("buckets").filter(lambda x: len(x) > 1)


arxiv_data_filtered["buckets"] = arxiv_data_filtered["buckets"].apply(
    lambda x: eval(x)
)

test_split = 0.1

# Initial train and test split.
train_df, test_df = train_test_split(
    arxiv_data_filtered,
    test_size=test_split,
    stratify=arxiv_data_filtered["buckets"].values,
)

val_df = test_df.sample(frac=0.5)
test_df.drop(val_df.index, inplace=True)

print(f"Number of rows in training set: {len(train_df)}")
print(f"Number of rows in validation set: {len(val_df)}")
print(f"Number of rows in test set: {len(test_df)}")

train_df = train_df[train_df["buckets"].notnull()]

buckets = tf.ragged.constant(train_df["buckets"].values)

lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
lookup.adapt(buckets)

vocab = lookup.get_vocabulary()


def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)


print("Vocabulary:\n")
print(vocab)

sample_label = train_df["buckets"].iloc[0]
print(f"Original label: {sample_label}")

label_binarized = lookup([sample_label])
print(f"Label-binarized representation: {label_binarized}")

print(train_df["desc"].apply(lambda x: len(str(x).split())).describe())

max_seqlen = 65
batch_size = 37
padding_token = "<pad>"
auto = tf.data.AUTOTUNE


def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["buckets"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices((dataframe["desc"].values, label_binarized))
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)


train_dataset = make_dataset(train_df, is_train=True)
validation_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)

text_batch, label_batch = next(iter(train_dataset))

for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text}")
    print(f"Label(s): {invert_multi_hot(label[0])}")
    print(" ")

vocabulary = set()
train_df["desc"].str.lower().str.split().apply(vocabulary.update)
vocabulary_size = len(vocabulary)
print(vocabulary_size)


text_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf")

with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

train_dataset = train_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
validation_dataset = validation_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
test_dataset = test_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)


# Define the model
def make_model():
    shallow_mlp_model = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(lookup.vocabulary_size(), activation="sigmoid"),
        ]
    )
    return shallow_mlp_model


# Train the model
epochs = 40

shallow_mlp_model = make_model()
shallow_mlp_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"]
)

history = shallow_mlp_model.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)


_, binary_acc = shallow_mlp_model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(binary_acc * 100, 2)}%.")


model_for_inference = keras.Sequential([text_vectorizer, shallow_mlp_model])

# Create a small dataset for demoing inference.
inference_dataset = make_dataset(test_df.sample(200), is_train=False)
print("Length of inference dataset", len(inference_dataset))
text_batch, label_batch = next(iter(inference_dataset))
predicted_probabilities = model_for_inference.predict(text_batch)

# Perform inference.
for i, text in enumerate(text_batch[:30]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text}")
    print(f"Label(s): {invert_multi_hot(label[0])}")

    predicted_proba = [proba for proba in predicted_probabilities[i]]
    #print("All probs ", predicted_proba)
    top_3_labels = [
        x
        for _, x in sorted(
            zip(predicted_probabilities[i], lookup.get_vocabulary()),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ][:3]
    print(f"Predicted Label(s): ({', '.join([label for label in top_3_labels])})")
    print(" ")