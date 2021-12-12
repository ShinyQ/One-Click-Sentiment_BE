import regex
import string
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from keras.preprocessing.sequence import pad_sequences

BAKU = pd.read_csv("app/dataset/Kamu-Alay.csv")
BAKU = BAKU.set_index("kataAlay")["kataBaik"].to_dict()


def tokenizing(text):
    text = regex.sub(r'http\S+', '', text)
    text = regex.sub('(@\w+|#\w+)', '', text)
    text = regex.sub('<.*?>', '', text)
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    text = regex.sub('[^a-zA-Z]', ' ', text)
    text = regex.sub("\n", " ", text)
    text = regex.sub(r'(\w)(\1{2,})', r"\1", text)
    text = regex.sub(r"\b[a-zA-Z]\b", "", text)
    text = regex.sub('(s{2,})', ' ', text)

    return text


def replace_at_to_User(text: str) -> str:
    return regex.sub("@[a-zA-Z0-9]+", "", text)


def remove_tanda(text: str) -> str:
    text = regex.sub("[!\"#%$&\'@()*+,-./:;<=>?[\\]^_`{|}~]+", "", text)
    return text


def remove_links(text: str) -> str:
    text = regex.sub("\S*:\S+", "", text)
    return text


def modified_has_tag(text: str) -> str:
    text = regex.sub("#", "", text).rstrip()
    return text


def stemming_data(text: str) -> str:
    return StemmerFactory().create_stemmer().stem(text)


def map_to_baku(text, baku):
    text_copy = text.split()
    for i in range(len(text_copy)):
        if text_copy[i] in baku:
            text_copy[i] = baku[text_copy[i]]

    return " ".join(text_copy)


def preprocessing_data(text):
    text = text.lower()
    text = tokenizing(text)
    text = map_to_baku(text, BAKU)
    text = remove_links(text)
    text = replace_at_to_User(text)
    text = modified_has_tag(text)
    text = remove_tanda(text)
    text = stemming_data(text)

    return text


def convert_input_to_sequences(text, max_length, model, tokenizer):
    text = preprocessing_data(text)
    sequence = tokenizer.texts_to_sequences([text])
    sequence_pad = pad_sequences(sequence, maxlen=max_length, padding="post")

    return text, (model.predict(sequence_pad))
