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

### 1 Эксперимент (original-model-1)
Делаю 1 эпоху запуска обычной модели, с обычным лоссом, когда мы просто на основе фич с последнего слоя предсказываем токены.
Иными словами "классическая" постановка задачи языкового моделирования.

### 2 Эксперимент (simple-classifiers)
Гипотеза: Если добавить классификатор после каждого трансформер блока, тогда лосс упадет.