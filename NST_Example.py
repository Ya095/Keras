#----------------------------------------
# Перенос стилей (Neural Style Transfer)
#----------------------------------------

# что бы убрать ошибку в консоле про использование gpu
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as  plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image


img = np.asarray(Image.open('images/img.jpeg'), dtype='uint8')
img_style = np.asarray(Image.open('images/img_style.jpeg'), dtype='uint8')

# преобразование в нужный для VGG-19 формат
x_img = keras.applications.vgg19.preprocess_input(np.expand_dims(img, axis=0))
x_style = keras.applications.vgg19.preprocess_input(np.expand_dims(img_style, axis=0))


def deprocess_img(processed_img):
    """ обратное преобразование из BGR в RGB """
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Входные данные должны быть вида: "
                            "[1, height, width, channel] или [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Неверный ввод для обработки изображения")

    # добавляем нужные значения к цветовым компонентам
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68

    x = x[:,:,::-1] # меняем местами цветовые компоненты
    x = np.clip(x, 0, 255).astype('uint8') # меняем значения -> 0 < x < 255
    return x

# выделяем слой из VGG19 для оценки потерь контента изображения
content_layer = ['block5_conv2']

# выделяем слои для оценки потерь стилизации изображения
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layer = len(content_layer)
num_style_layers = len(style_layers)

# загружаем НС VGG19 (без полносвязной НС в конце; с предобученными весами)
vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False # веса нельзя менять

# выделяем нужные слои выхода VGG19 (название можно увидеть в структуре сети)
style_outputs = [vgg.get_layer(name=name).output for name in style_layers]
content_output = [vgg.get_layer(name=name).output for name in content_layer]
model_outputs = style_outputs + content_output # объединяем

print(vgg.input)
for m in model_outputs:
    print(m)
print()

# модель с одним входом и несколькими нужными выходами
model = keras.models.Model(vgg.input, model_outputs)
model.summary()


def get_feature_representations(model):
    """ возвращает карту признаков для стилей и контента """
    style_outputs = model(x_style)
    content_outputs = model(x_img)

    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


def get_content_loss(base_content, target):
    """ вычисление потерь по контенту изображения
       - base_content -> тензор на последнем слое НС исходного изображения
       - target -> тензор на последнем слое НС полученного изображения
     """
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    """ вычисление матрицы Грамма для передаваемого тензора """
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_loss_style(base_style_map, gram_target_style):
    """ вычисление рассогласования по стилям для конкретного слоя СНС
        - base_style_map -> исходное изображение стилей
        - gram_target_style -> матрица Грамма для нужного слоя (изобр. стилей)
    """
    gram_style = gram_matrix(base_style_map)
    return tf.reduce_mean(tf.square(gram_style - gram_target_style))

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    """ вычисляются все потери
        - model -> модель НС
        - loss_weights -> веса для контента и стилей
        - init_image -> формируемое изображение
        - gram_style_features -> матрица Грамма для изображения стилей
        - content_features -> карта признаков для контента (с last слоя НС)
    """
    style_weight, content_weight = loss_weights

    # пропускаем исходное изображение через НС
    # на выходе получаем зн. на каждом нужном сверточном слое
    model_outputs = model(init_image)

    # сформированные карты признаков, поле прогона через НС
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    # величины потерь для стиля и контента
    style_score = 0
    content_score = 0

    # веса для потерь стилей для каждого слоя (важность)
    weights_per_style_layer = 1.0 / float(num_style_layers)

    # суммируем квадраты рассогласований для каждого слоя
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weights_per_style_layer * get_loss_style(comb_style[0], target_style)

    # аналогично для контентного слоя
    weights_per_content_layer = 1.0 / float(num_content_layer)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score  += weights_per_content_layer * get_content_loss(comb_content, target_content)

    style_score *= style_weight
    content_score *= content_weight

    # общий критерий качества
    loss = style_score + content_score
    return loss, style_score, content_score


iterations = 100
content_weight = 1e3
style_weight = 1e-2

# пропускаем через НС 2 начальных изображения и получаем карты признаков
style_features, content_features = get_feature_representations(model)
# матрица Грамма для изображения со стилями
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

# init_image - начальное формируемое изображение (в понятный ts формат)
init_image = np.copy(x_img)
init_image = tf.Variable(init_image, dtype=tf.float32)

opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.99, epsilon=1e-1)
iter_count = 1
# наименьшие потери, наилучшее изображение
best_loss, best_img = float('inf'), None
loss_weights = (style_weight, content_weight)

cfg = {
    'model': model,
    'loss_weights': loss_weights,
    'init_image': init_image,
    'gram_style_features': gram_style_features,
    'content_features': content_features
}

norm_means = np.array([103.939, 116.779, 123.68])
min_vals = -norm_means
max_vals = 255 - norm_means
images = [] # тут будут все сформированные за время работы алгоритма изобр.

# запуск алгоритма град. спуска
for i in range(iterations):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)

    total_loss = all_loss[0]
    grads = tape.gradient(total_loss, init_image)

    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    # ограничиваем мин и макс значениями
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)

    # проверяем, для какого изображения получились наименьшие потери
    if loss < best_loss:
        best_loss = loss
        best_img = deprocess_img(init_image.numpy())

        plot_img = deprocess_img(init_image.numpy())
        images.append(plot_img)

        print('Iteration: {}'.format(i) )


plt.imshow(best_img)
plt.show()
print(best_loss)

image = Image.fromarray(best_img.astype('uint8'), 'RGB')
image.save('images/outs/out.jpg')