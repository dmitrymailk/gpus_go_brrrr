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

## Experiment runs

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