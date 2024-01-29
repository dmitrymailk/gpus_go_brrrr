## [Experiment runs, wandb](https://wandb.ai/dimweb/llm_lora/workspace?workspace=user-dimweb)


### 1 Эксперимент (original-model-pythia-410m-1)

Простой лосс языкового моделирования.
- PARAMETERS 405_334_016

### 2 Эксперимент (linear-lora-8-2)
Заменить все оригинальные слои в attention на линейную лору 8
- PARAMETERS 330_623_168

### 3 Эксперимент (nonlinear-lora-8-3)
Заменить все оригинальные слои в attention на нелинейную лору 8
- PARAMETERS 330_623_168

### 4 Эксперимент (linear-lora-32-4)
Заменить все оригинальные слои в attention на линейную лору 32
- PARAMETERS 332_983_040

### 5 Эксперимент (linear-lora-256-5)
Заменить все оригинальные слои в attention на линейную лору 256
- PARAMETERS 355_008_512

### 6 Эксперимент (no-attention-6)
Убираю полностью attention.

- PARAMETERS 304_572_416

### 7 Эксперимент (linear-lora-8-all-attention-7)
Заменить все линейные слои в attention на lora 8.

- PARAMETERS 305_850_752

### 8 Эксперимент (linear-lora-8-no-dense-8)
Заменить все линейные слои в attention на lora 8. И убрать линейный слой после вычисления multihead attention.

- PARAMETERS 305_432_768

### 9 Эксперимент (linea-mlp-9)
Убрал нелинейность из MLP блока.

- PARAMETERS 305_432_768

### 10 Эксперимент (nonlinear-lora-8-mlp-10)
Изменил MLP на lora c r=8. И нелинейностью как в исходном блоке.

- PARAMETERS 104_401_280

### 11 Эксперимент (nonlinear-lora-768-mlp-11)
Изменил MLP на lora c r=768. И нелинейностью как в исходном блоке.

- PARAMETERS 141_775_040

### 12 Эксперимент (partial-linear-lora-8-attention-12)
На 1 слое идет полный attention, на следующих lora 8.
- PARAMETERS 308_545_720

### 13 Эксперимент (6-layers-partial-linear-lora-8-attention-13)
До 6 слоя идет полный attention, на следующих lora 8.
- PARAMETERS 324_110_480

### 14 Эксперимент (3-layers-partial-linear-lora-8-attention-14)
До 3 слоя идет полный attention, на следующих lora 8.
- PARAMETERS 314_771_624

### 15 Эксперимент (2-layers-partial-linear-lora-8-attention-15)
До 2 слоя идет полный attention, на следующих lora 8.
- PARAMETERS 311_658_672

### 16 Эксперимент (3-layers-partial-linear-lora-6-nonlinear-lora-8-attention-16)
До 3 слоя идет полный attention, на следующих lora 8.
До 6 слоя идет обычный MLP, потом nonlinear lora 8.

- PARAMETERS 163_998_008

### 17 Эксперимент (3-layers-partial-linear-lora-12-nonlinear-lora-8-attention-17)
До 3 слоя идет полный attention, на следующих lora 8.
До 12 слоя идет обычный MLP, потом nonlinear lora 8.

- PARAMETERS 214_255_880

### 18 Эксперимент (3-layers-partial-linear-lora-12-nonlinear-lora-8-256-attention-18)
До 3 слоя идет полный attention, на следующих lora 8.
До 12 слоя идет обычный MLP, потом nonlinear lora 256.

- PARAMETERS 214_255_880

### 19 Эксперимент (original-model-pythia-2.8b-19)
Дефолтная модель EleutherAI/pythia-2.8b.

- PARAMETERS 2_775_208_960

### 20 Эксперимент (3-layers-partial-linear-lora-8-pythia-2.8b-20)
До 3 слоя идет полный attention, на следующих lora 8.
Модель EleutherAI/pythia-2.8b.

- PARAMETERS 1_997_624_552

### 21 Эксперимент (6-layers-partial-linear-lora-8-pythia-2.8b-21)
До 6 слоя идет полный attention, на следующих lora 8.
Модель EleutherAI/pythia-2.8b.

- PARAMETERS 2_056_361_168
