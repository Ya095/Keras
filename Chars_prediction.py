#------------------------------------
# Прогнозирование символов (RNN)
#------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re
from keras.layers import Dense, SimpleRNN, Input
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer


with open('text_files/train_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '') # убираем первый невидимый символ
    text = re.sub(r'[^А-я ]', '', text) # заменяем все символы кроме кириллицы на пустые

# парсим текст как последовательность символов
num_charset = 34 # 33 буквы + пробел
tokenizer = Tokenizer(num_words=num_charset, char_level=True)
tokenizer.fit_on_texts([text]) # формируем токены на основе частотности в тексте

inp_chars = 6
data = tokenizer.texts_to_matrix(text) # преобразуем исходный текст в массив one-hot векторов
n = data.shape[0] - inp_chars # так предсказываем по 3 символам - четвертый
# print(data.shape) # -> (3528, 34)

# формируем входной тензор (потом смещаемся на 1 вектор) - и так на каждом шаге
X = np.array([data[i : i + inp_chars, :] for i in range(n)])
# коллекция из one-hot векторов, для которых надо построить прогноз
Y = data[inp_chars:]


model = Sequential()
model.add(Input((inp_chars, num_charset)))
model.add(SimpleRNN(128, activation='tanh')) # рекуррентный слой на 128 нейронов
model.add(Dense(num_charset, activation='softmax'))
# model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adam')
hist = model.fit(X, Y, batch_size=32, epochs=100)


def build_phrase(inp_srt, str_len=50):
    """
    Строим фразу на основе прогнозных значений
        inp_srt -> начальные символы
        str_len -> какую длину строки хотим получить
    """
    for i in range(str_len):
        x = []
        for j in range(i, i+inp_chars):
            x.append(tokenizer.texts_to_matrix((inp_srt[j]))) # преобразуем символы в one-hot вектор

        x = np.array(x)
        inp = x.reshape(1, inp_chars, num_charset) # делаем нужный формат для подачи в НС
        pred = model.predict(inp) # предсказываем ОНЕ 4 символа
        d = tokenizer.index_word[pred.argmax(axis=1)[0]] # ответ в символьном представлении

        inp_srt += d

    return inp_srt


res = build_phrase('нейрон')
print(res)