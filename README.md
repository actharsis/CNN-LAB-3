# Изучение влияние параметра “темп обучения” на процесс обучения нейронной сети на примере решения задачи классификации Food-101 с использованием техники обучения Transfer Learning
## Фиксированный темп обучения в сочетании с Transfer Learning
Файл:
```
CNN-food-101-master/transfer_train.py
```
Архитектура:
```python
inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
model = EfficientNetB0(include_top=False, weights="imagenet", classes=NUM_CLASSES, input_tensor=inputs)
model.trainable = False
x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
return tf.keras.Model(inputs=inputs, outputs=outputs)
```
![legend1](https://user-images.githubusercontent.com/24518594/115959624-c866ab00-a515-11eb-8171-506fd726d86a.png)

Метрика качества:
![gr1](https://github.com/actharsis/lab3/blob/main/graphs/epoch_categorical_accuracy_const_lr.svg)

Функция потерь:
![gr2](https://github.com/actharsis/lab3/blob/main/graphs/epoch_loss_const_lr.svg)
## Косинусное затухание
Файл:
```
CNN-food-101-master/train_cosine_decay.py
```
```python
def decayed_learning_rate(step):
  step = min(step, decay_steps)
  cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
  decayed = (1 - alpha) * cosine_decay + alpha
  learning_rate = initial_learning_rate * decayed

  tf.summary.scalar('learning rate', data=learning_rate, step=step)
  return learning_rate
```

Метрика качества:

Функция потерь:

График темпа обучения:
## Косинусное затухание с перезапусками

## Анализ результатов
