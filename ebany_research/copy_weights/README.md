https://wandb.ai/dimweb/copy_weights?workspace=user-dimweb

### 1 Эксперимент (2-layers-fullfinetune)
Обучение происходит на 2 слоях последовательно. Обычное обучение GPT.

### 2 Эксперимент (copy-weights-30-steps-epoch-0)
Копирую веса с первого на второй слой после 30 шагов на первой эпохе. Далее учу как обычно.
Результаты: Сначала лосс слишком сильно подскочил и потом не опускался ниже дефолтного. Нужно попробовать сократить до 2 шагов.

### 3 Эксперимент (copy-weights-2-steps-epoch-0)
Копирую веса с первого на второй слой после 2 шагов на первой эпохе. Далее учу как обычно.
Результаты: Лосс слегка выше чем из эксперимента 1. нужно попробовать сократить до 1 шагов.

### 4 Эксперимент (copy-weights-1-steps-epoch-0)
Копирую веса с первого на второй слой после 1 шагов на первой эпохе. Далее учу как обычно.
Результаты: Лосс получился выше чем из эксперимента 1.  Стоит попбовать тоже самое с 3 слоями.

### 5 Эксперимент (copy-weights-3-steps-epoch-0)
Копирую веса с первого на второй слой после 1 шагов на первой эпохе. Далее учу как обычно.
Результаты: Лосс получился ниже из эксперимента 1. Но их разница Фактически незначительна. Стоит попбовать тоже самое с 3 слоями.

### 6 Эксперимент (6-layers-fullfinetune)
Обучение происходит на 6 слоях последовательно. Обычное обучение GPT.
Результаты: Лосс стал ниже всех. 

### 7 Эксперимент (6-layers-copy-step-1)
Я сначала обучаю первый слой, делаю один шаг. Затем копирую данные веса на второй слой и делаю следующий шаг. И тд
Результаты: лосс намного выше чем при обычном обучении. 

### 8 Эксперимент (6-layers-copy-step-2)
Я сначала обучаю первый слой, делаю 2 шага. Затем копирую данные веса на второй слой и делаю следующие 2 шага. И тд
Результаты: лосс намного выше чем при обычном обучении. 