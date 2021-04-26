# Изучение влияние параметра “темп обучения” на процесс обучения нейронной сети на примере решения задачи классификации Food-101 с использованием техники обучения Transfer Learning
## Фиксированный темп обучения в сочетании с Transfer Learning
Файл: `CNN-food-101-master/transfer_train.py`

Архитектура:
```python
inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
model = EfficientNetB0(include_top=False, weights="imagenet", classes=NUM_CLASSES, input_tensor=inputs)
model.trainable = False
x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
return tf.keras.Model(inputs=inputs, outputs=outputs)
```
Вариации learning_rate: 0.01, 0.001, 0.0001

![legend1](https://user-images.githubusercontent.com/24518594/115959624-c866ab00-a515-11eb-8171-506fd726d86a.png)

Метрика качества:
![gr1](https://github.com/actharsis/lab3/blob/main/graphs/epoch_categorical_accuracy_const_lr.svg)

Функция потерь:
![gr2](https://github.com/actharsis/lab3/blob/main/graphs/epoch_loss_const_lr.svg)
## Косинусное затухание
Файл: `CNN-food-101-master/train_cosine_decay.py`

```python
def decayed_learning_rate(step):
  step = min(step, decay_steps)
  cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
  decayed = (1 - alpha) * cosine_decay + alpha
  learning_rate = initial_learning_rate * decayed

  tf.summary.scalar('learning rate', data=learning_rate, step=step)
  return learning_rate
```
Был использован initial_learning_rate=0.01 и следующие значения decay_steps: 1000, 50, 10

![Legend2](https://user-images.githubusercontent.com/24518594/116000077-70a07080-a5f7-11eb-870f-9204a4fa18b6.png)

Метрика качества на валидации:
![gr3](https://github.com/actharsis/lab3/blob/main/graphs/epoch_categorical_accuracy_cosine.svg)

Функция потерь на валидации:
![gr4](https://github.com/actharsis/lab3/blob/main/graphs/epoch_loss_cosine.svg)

График темпа обучения:
![gr5](https://github.com/actharsis/lab3/blob/main/graphs/learning%20rate_cosine.svg)

Также были попытки менять initial_learning_rate(от 0.0001 до 0.5), при высоких значениях функция потерь была очень высокой(около 20 к 5 эпохе) и показатели точности всего лишь 55%. При маленьких значениях initial_learning_rate функция ошибок к 10 эпохе была не выше 3, но несмотря на это итоговая точность на валидации была ниже, чем при фиксированном темпе обучения, обучение с явно неудачными параметрами прерывалось к 10-15 эпохе, поэтому в итоге было принято решение использовать начальный темп 0.001 и в дальнешем варьировать лишь decay_steps.
## Косинусное затухание с перезапусками
Файл: `CNN-food-101-master/train_cosine_restarts.py`
```python
learning_rate = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate, first_decay_steps)
```
initial_learning_rate = 0.001, first_decay_steps: 10, 100

![Legend3](https://user-images.githubusercontent.com/24518594/116113261-018f4e80-a6c1-11eb-8d6a-7385f891cce7.png)

Метрика качества на валидации:
![gr6](https://github.com/actharsis/lab3/blob/main/graphs/cosine_restart/epoch_categorical_accuracy.svg)

Функция потерь на валидации:
![gr7](https://github.com/actharsis/lab3/blob/main/graphs/cosine_restart/epoch_loss.svg)

![Screenshot_1](https://user-images.githubusercontent.com/24518594/116113386-21bf0d80-a6c1-11eb-9ec8-1ae97b5c0a5f.png)
![Screenshot_3](https://user-images.githubusercontent.com/24518594/116113400-24216780-a6c1-11eb-927d-77d58719fcea.png)

График темпа обучения:
![gr8](https://github.com/actharsis/lab3/blob/main/graphs/cosine_restart/epoch_learning_rate.svg)

Как и в случае с косинусным затуханием, изменение initial_learning_rate давало плохие результаты: потери в точности больше 5% на валидации в случае с большими значениями и около 3% в случае со слишком маленькими, изменение first_decay_steps слабо сказывалось на результатах, поэтому опять было принято решение зафиксировать initial_learning_rate на 0.001
## Анализ результатов
При изучении фиксированного темпа обучения оптимальным оказался темп 0.0001, он показал наивысшую точность - 66% на валидации. В случае с косинусным затуханием оптимальной комбинацией параметров был initial_learning_rate=0.01 и decay_steps=10, несмотря на то, что при таких параметрах темп обучения после 10 эпохи равен нулю, точность на валидации составила 67.1%, что на 1.1% больше, чем при фиксированном темпе. При использовании косинусного затухание с перезапусками оптимальными подобранными параметрами оказались initial_learning_rate=0.01 и first_decay_steps=100, максимальная точность на валидации составила 67.36%: на 1.36% больше, чем наилучший результат при фиксированном темпе обучения, таким образом эта политика оказалась наиболее эффективной.
