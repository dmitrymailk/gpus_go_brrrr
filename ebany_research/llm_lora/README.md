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
До 6 слоя идет обычный MLP, потом nonlinear lora 8. [НЕПРАВИЛЬНО]

- PARAMETERS 163_998_008

### 17 Эксперимент (3-layers-partial-linear-lora-12-nonlinear-lora-8-attention-17)
До 3 слоя идет полный attention, на следующих lora 8.
До 12 слоя идет обычный MLP, потом nonlinear lora 8. [НЕПРАВИЛЬНО]

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

### 22 Эксперимент (6-layers-partial-linear-lora-8-20-layers-LoraMLP-8-22)
До 6 слоя идет полный attention, на следующих lora 8.
До 20 слоя идет полный MLP, на следующих lora 16.
Модель EleutherAI/pythia-410m.
-  PARAMETERS 291_211_536

### 23 Эксперимент (6-layers-partial-linear-lora-8-12-18-layers-LoraMLP-8-23)
До 6 слоя идет полный attention, на следующих lora 8.
До pos >= 12 and pos <= 18 идут lora 16 MLP. 7 слоев
Модель EleutherAI/pythia-410m.
- PARAMETERS 266_537_328

По итогу совпало с оригиналом.

### 24 Эксперимент (6-layers-partial-linear-lora-8-12-18-layers-LoraMLP-8-24)
До 6 слоя идет полный attention, на следующих lora 8.
До pos >= num_hidden_layers // 2 and pos <= num_hidden_layers // 2 + 6 идут lora 16 MLP. 7 слоев.

Модель EleutherAI/pythia-2.8b.
- PARAMETERS 1_692_226_992

Лосс совпал с исходной моделью.

### 25 Эксперимент (freeze-embeddings-25)
Модель EleutherAI/pythia-410m.
Замораживаю все эмбеддинги. Все остальное в модели остается неизменным.
- PARAMETERS 353822720
Лосс намного выше

### 26 Эксперимент (lora-embeddings-26)
Модель EleutherAI/pythia-410m.
Делаю собственные эмбеддинги с lora.
```python
class CustomEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Tensor | None = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype,
        )
        self.weight = None
        factory_kwargs = {"device": device, "dtype": dtype}
        r = 256 * 2
        self.A = Parameter(
            torch.empty((num_embeddings, r), **factory_kwargs),
            requires_grad=not _freeze,
        )
        self.B = Parameter(
            torch.empty((r, embedding_dim), **factory_kwargs),
            requires_grad=not _freeze,
        )
        self.B.data.normal_(mean=0.0, std=0.02)
        self.A.data.normal_(mean=0.0, std=0.02)

    def forward(self, input: Tensor) -> Tensor:
        weight = self.A @ self.B
        return F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
```
- PARAMETERS 405334016
Лосс выше, а параметров меньше всего на 10 миллионов. Эмбеддинги лучше не трогать.
