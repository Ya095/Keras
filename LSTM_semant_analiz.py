#----------------------------------------
# Семантический анализ изображений
#----------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


with open('text_files/train_data_true.txt', 'r', encoding='utf-8') as f:
    texts_true = f.readlines()
    texts_true[0] = texts_true[0].replace('\ufeff', '')

with open('text_files/train_data_false.txt', 'r', encoding='utf-8') as f:
    texts_false = f.readlines()
    texts_false[0] = texts_false[0].replace('\ufeff', '')


texts = texts_true + texts_false
count_true = len(texts_true)
count_false = len(texts_false)
total_lines = count_true + count_false

# разбиваем на отдельные слова
maxWordsCount = 1000
tokenizer = Tokenizer(num_words=maxWordsCount, lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts(texts)

# dct = list(tokenizer.word_counts.items()) # -> [[("думайте", 2), ("и", 34), ...]]
# dct = list(tokenizer.word_index.items()) # -> [[("не", 1), ("как", 2), ("итого", 3), ...]]
# print([dct[:5]])

# преобразование текста (разбит на слова) в последовательность чисел
max_text_len = 10
data = tokenizer.texts_to_sequences(texts)
data_pad = pad_sequences(data, maxlen=max_text_len)

# формирование обучающей выборки
x = data_pad
y = np.array([[1,0]]*count_true + [[0,1]]*count_false)

# перемешиваем наблюдения для лучшего обучения НС
indexes = np.random.choice(x.shape[0], size=x.shape[0], replace=False)
x = x[indexes]
y = y[indexes]

# создание модели
model = Sequential()
model.add(Embedding(maxWordsCount, 128, input_length=max_text_len))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(2, activation='softmax'))
# model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))
hist = model.fit(x, y, batch_size=32, epochs=50)


def seq_to_text(list_of_indexes):
    """ Преобразовывает индексы в текст """
    # ('и', 2) -> (2, 'и')
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    words = [reverse_word_map.get(letter) for letter in list_of_indexes]
    return words


# не делаем fit_on_texts, тк словарь слов уже сформирован и эти слова уже есть в словаре
t = "Оставайтесь на позитиве".lower()
data = tokenizer.texts_to_sequences([t])
data_pad = pad_sequences(data, maxlen=max_text_len)
print(seq_to_text(data[0]))

res = model.predict(data_pad)
print(res, np.argmax(res), sep='\n')