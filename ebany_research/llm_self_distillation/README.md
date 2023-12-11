- https://arxiv.org/pdf/1909.10351.pdf
- https://github.com/luanyunteng/pytorch-be-your-own-teacher
- https://www.computer.org/csdl/journal/tp/2022/08/09381661/1s4kVUKSRfq - реплецировать это для pythia и wikitext-2-raw-v1
- https://arxiv.org/pdf/2210.11610.pdf
- [Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation](https://arxiv.org/pdf/1905.08094.pdf)
- [TinyML and Efficient Deep Learning Computing ](https://hanlab.mit.edu/courses/2023-fall-65940)
- [Week 9: Efficient model inference](https://github.com/mryab/efficient-dl-systems/tree/main/week09_compression)

Если для resnet классификация классов картинок, то для LLM классификация токенов, вот и всё.

Идеи:
- сделать так чтобы все блоки трансформера стремились получить веса следующего. 

## [Experiment runs, wandb](https://wandb.ai/dimweb/llm_self_distillation)

### 1 Эксперимент (original-model-pythia-70m-1)
Делаю 1 эпоху запуска обычной модели, с обычным лоссом, когда мы просто на основе фич с последнего слоя предсказываем токены.
Иными словами "классическая" постановка задачи языкового моделирования.

### 2 Эксперимент (simple-classifiers-pythia-70m-2)
Гипотеза: Если добавить классификатор после каждого трансформер блока, тогда лосс упадет.
Результаты:
```text
wandb:                   epoch 3
wandb:               eval_loss 6.69411
wandb:              perplexity 807.63745
wandb:                    step 584
wandb: total_train_loss_step_0 6.74064
wandb: total_train_loss_step_1 6.70205
wandb: total_train_loss_step_2 6.68686
wandb: total_train_loss_step_3 6.69178
wandb: total_train_loss_step_4 6.68755
wandb: total_train_loss_step_5 7.12703
wandb:              train_loss 6.68755
wandb:         train_loss_step 6.59375
wandb:       train_loss_step_0 6.71875
wandb:       train_loss_step_1 6.65625
wandb:       train_loss_step_2 6.625
wandb:       train_loss_step_3 6.625
wandb:       train_loss_step_4 6.59375
wandb:       train_loss_step_5 7.0625
```

Добавление классификаторов действительно помогло уменьшить лосс. Однако почему-то на последнем блоке total_train_loss_step_5 прослеживается обратная ситуация с ошибкой. Она стала не самой маленькой, а самой высокой.

Планы: 
- Нужно запустить на других размерах моделей и посмотреть на результат.

### 3 Эксперимент (simple-classifiers-pythia-70m-3)
Вопрос: если использовать 1 классификатор на основе всех фич, что будет? 
Результат: 
```text
wandb:                   epoch 3
wandb:               eval_loss 6.7159
wandb:              perplexity 825.4243
wandb:                    step 584
wandb: total_train_loss_step_0 6.82181
wandb: total_train_loss_step_1 6.72977
wandb: total_train_loss_step_2 6.71506
wandb: total_train_loss_step_3 6.70896
wandb: total_train_loss_step_4 6.70275
wandb: total_train_loss_step_5 7.79811
wandb:              train_loss 6.70275
wandb:         train_loss_step 6.625
wandb:       train_loss_step_0 6.75
wandb:       train_loss_step_1 6.65625
wandb:       train_loss_step_2 6.625
wandb:       train_loss_step_3 6.625
wandb:       train_loss_step_4 6.625
wandb:       train_loss_step_5 7.78125
```
Качество получилось почти такое же как и со множественными классификаторами. Странно.

### 4 Эксперимент (original-model-pythia-410m-4)

Повторяет код из `1 Эксперимент`. Просто увеличил размер модели.
Проверка оригинальной модели на сходимость.
Результаты:
```text
wandb: Run summary:
wandb:           epoch 3
wandb:       eval_loss 6.25
wandb:      perplexity 518.01282
wandb:            step 584
wandb:      train_loss 6.03489
wandb: train_loss_step 5.96875
```

### 5 Эксперимент (simple-classifiers-pythia-410m-5)
Повторяет код из `2 Эксперимент`. Просто увеличил размер модели.
Результаты:
```text
wandb: Run summary:
wandb:                    epoch 3
wandb:                eval_loss 6.23522
wandb:               perplexity 510.4119
wandb:                     step 1168
wandb:  total_train_loss_step_0 6.72925
wandb:  total_train_loss_step_1 6.51799
wandb:  total_train_loss_step_2 6.43458
wandb:  total_train_loss_step_3 6.37503
wandb:  total_train_loss_step_4 6.33826
wandb:  total_train_loss_step_5 6.30334
wandb:  total_train_loss_step_6 6.2805
wandb:  total_train_loss_step_7 6.26505
wandb:  total_train_loss_step_8 6.24968
wandb:  total_train_loss_step_9 6.23811
wandb: total_train_loss_step_10 6.22338
wandb: total_train_loss_step_11 6.21305
wandb: total_train_loss_step_12 6.20051
wandb: total_train_loss_step_13 6.18994
wandb: total_train_loss_step_14 6.17797
wandb: total_train_loss_step_15 6.16959
wandb: total_train_loss_step_16 6.16552
wandb: total_train_loss_step_17 6.1536
wandb: total_train_loss_step_18 6.14768
wandb: total_train_loss_step_19 6.13788
wandb: total_train_loss_step_20 6.13571
wandb: total_train_loss_step_21 6.12487
wandb: total_train_loss_step_22 6.11437
wandb: total_train_loss_step_23 6.3607
wandb:               train_loss 6.11437
wandb:          train_loss_step 6.03125
wandb:        train_loss_step_0 6.6875
wandb:        train_loss_step_1 6.4375
wandb:        train_loss_step_2 6.375
wandb:        train_loss_step_3 6.3125
wandb:        train_loss_step_4 6.25
wandb:        train_loss_step_5 6.28125
wandb:        train_loss_step_6 6.25
wandb:        train_loss_step_7 6.15625
wandb:        train_loss_step_8 6.21875
wandb:        train_loss_step_9 6.125
wandb:       train_loss_step_10 6.125
wandb:       train_loss_step_11 6.09375
wandb:       train_loss_step_12 6.15625
wandb:       train_loss_step_13 6.125
wandb:       train_loss_step_14 6.09375
wandb:       train_loss_step_15 6.09375
wandb:       train_loss_step_16 6.0625
wandb:       train_loss_step_17 6.09375
wandb:       train_loss_step_18 6.09375
wandb:       train_loss_step_19 6.0625
wandb:       train_loss_step_20 6.09375
wandb:       train_loss_step_21 6.03125
wandb:       train_loss_step_22 6.03125
wandb:       train_loss_step_23 6.3125
```

На более большой модели, классификаторы на фичи после каждого блока дают более плохой результат.
Примерно на 6.11437 - 6.03489 = 0.07948.
Гипотезы: может быть это связано с батчсайзом

### 6 Эксперимент (simple-classifiers-pythia-410m-6)

```text
wandb: Run summary:
wandb:                    epoch 3
wandb:                eval_loss 6.32806
wandb:               perplexity 560.06936
wandb:                     step 584
wandb:  total_train_loss_step_0 6.77335
wandb:  total_train_loss_step_1 6.60103
wandb:  total_train_loss_step_2 6.54852
wandb:  total_train_loss_step_3 6.51593
wandb:  total_train_loss_step_4 6.49606
wandb:  total_train_loss_step_5 6.46843
wandb:  total_train_loss_step_6 6.4515
wandb:  total_train_loss_step_7 6.43927
wandb:  total_train_loss_step_8 6.42856
wandb:  total_train_loss_step_9 6.41934
wandb: total_train_loss_step_10 6.40017
wandb: total_train_loss_step_11 6.39224
wandb: total_train_loss_step_12 6.37856
wandb: total_train_loss_step_13 6.36723
wandb: total_train_loss_step_14 6.35312
wandb: total_train_loss_step_15 6.34573
wandb: total_train_loss_step_16 6.34032
wandb: total_train_loss_step_17 6.32744
wandb: total_train_loss_step_18 6.32099
wandb: total_train_loss_step_19 6.31247
wandb: total_train_loss_step_20 6.30931
wandb: total_train_loss_step_21 6.29975
wandb: total_train_loss_step_22 6.28867
wandb: total_train_loss_step_23 6.4977
wandb:               train_loss 6.28867
wandb:          train_loss_step 6.15625
wandb:        train_loss_step_0 6.71875
wandb:        train_loss_step_1 6.5
wandb:       train_loss_step_10 6.25
wandb:       train_loss_step_11 6.25
wandb:       train_loss_step_12 6.28125
wandb:       train_loss_step_13 6.28125
wandb:       train_loss_step_14 6.25
wandb:       train_loss_step_15 6.1875
wandb:       train_loss_step_16 6.1875
wandb:       train_loss_step_17 6.21875
wandb:       train_loss_step_18 6.21875
wandb:       train_loss_step_19 6.15625
wandb:        train_loss_step_2 6.46875
wandb:       train_loss_step_20 6.1875
wandb:       train_loss_step_21 6.15625
wandb:       train_loss_step_22 6.15625
wandb:       train_loss_step_23 6.40625
wandb:        train_loss_step_3 6.4375
wandb:        train_loss_step_4 6.40625
wandb:        train_loss_step_5 6.40625
wandb:        train_loss_step_6 6.375
wandb:        train_loss_step_7 6.3125
wandb:        train_loss_step_8 6.375
wandb:        train_loss_step_9 6.28125
```
Примерно на 6.28867 - 6.03489 = 0.25378
Вывод: увеличение батчсайза только ухудшило результат.

### 7 Эксперимент (simple-classifiers-pythia-410m-7)
Повторение эксперимента `simple-classifiers-pythia-70m-3`, но для более большой модели.
Результаты:
```text
wandb: Run summary:
wandb:                    epoch 3
wandb:                eval_loss 6.4375
wandb:               perplexity 624.84304
wandb:                     step 584
wandb:  total_train_loss_step_0 9.50594
wandb:  total_train_loss_step_1 7.01957
wandb:  total_train_loss_step_2 6.64142
wandb:  total_train_loss_step_3 6.53861
wandb:  total_train_loss_step_4 6.48308
wandb:  total_train_loss_step_5 6.44626
wandb:  total_train_loss_step_6 6.41918
wandb:  total_train_loss_step_7 6.39037
wandb:  total_train_loss_step_8 6.37122
wandb:  total_train_loss_step_9 6.35481
wandb: total_train_loss_step_10 6.33989
wandb: total_train_loss_step_11 6.32565
wandb: total_train_loss_step_12 6.30891
wandb: total_train_loss_step_13 6.29914
wandb: total_train_loss_step_14 6.29167
wandb: total_train_loss_step_15 6.28599
wandb: total_train_loss_step_16 6.28184
wandb: total_train_loss_step_17 6.27836
wandb: total_train_loss_step_18 6.2763
wandb: total_train_loss_step_19 6.27565
wandb: total_train_loss_step_20 6.27504
wandb: total_train_loss_step_21 6.27723
wandb: total_train_loss_step_22 6.2824
wandb: total_train_loss_step_23 7.15614
wandb:               train_loss 6.2824
wandb:          train_loss_step 6.15625
wandb:        train_loss_step_0 9.4375
wandb:        train_loss_step_1 6.9375
wandb:        train_loss_step_2 6.53125
wandb:        train_loss_step_3 6.46875
wandb:        train_loss_step_4 6.40625
wandb:        train_loss_step_5 6.34375
wandb:        train_loss_step_6 6.3125
wandb:        train_loss_step_7 6.28125
wandb:        train_loss_step_8 6.25
wandb:        train_loss_step_9 6.25
wandb:       train_loss_step_10 6.21875
wandb:       train_loss_step_11 6.21875
wandb:       train_loss_step_12 6.21875
wandb:       train_loss_step_13 6.1875
wandb:       train_loss_step_14 6.1875
wandb:       train_loss_step_15 6.15625
wandb:       train_loss_step_16 6.15625
wandb:       train_loss_step_17 6.15625
wandb:       train_loss_step_18 6.15625
wandb:       train_loss_step_19 6.15625
wandb:       train_loss_step_20 6.15625
wandb:       train_loss_step_21 6.1875
wandb:       train_loss_step_22 6.15625
wandb:       train_loss_step_23 7.125
```

Примерно на 6.2824 - 6.03489 = 0.24751.
Вывод: Данный метод почти никак не повлиял, в районе погрешности. Данный метод сработал на более маленькой модели, но 
на большой показал только деградацию. Необходимо применить метрики из статьи. `Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation`.

### 8 Эксперимент (simple-classifiers-pythia-70m-8)
Применение лосса с использованием 