## Низкоранговое сжатие модели без потерь 

#### Оригинальная модель
- Model: Open-Orca/Mistral-7B-OpenOrca
- 7_241_748_480

#### Теоретический минимум
Если все линейные слои заменить на lora размерности 16.
- 173_867_040

Скорее всего такой сценарий невозможен. Так как при тренировке без потерь можно убрать только attention после 6 слоя, а линейных слоев можно декомпозировать всего 7. Плюс декомпозировать голову при тоже не удается. С учетом этого предполагаемое сжатие составит. Это желаемый результат при котором целевые метрики никак не меняют свои значения.
-  4_935_372_800

Если заменить только attention слои
- 6_162_305_024

Из это следует что основное внимание стоит состредоточить на MLP слое. Потому что в них содержится наибольшее количество параметров.


### Experiment 1

```python
from torch.nn.parameter import Parameter


class LinearLora(torch.nn.Module):
    def __init__(self, in_dim=768, out_dim=768, r=16, bias=False):
        super().__init__()

        self.dense_h_to_4h = torch.nn.Linear(in_dim, r, bias=bias)

        self.dense_4h_to_h = torch.nn.Linear(r, out_dim, bias=bias)


    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)

        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class Lora(torch.nn.Module):
    def __init__(
        self,
        in_dim=768,
        out_dim=768,
        r=8,
        bias=True,
    ):
        super().__init__()

        self.A = Parameter(
            torch.empty(
                (in_dim, r),
            ),
            requires_grad=True,
        )
        self.B = Parameter(
            torch.empty(
                (r, out_dim),
            ),
            requires_grad=True,
        )
        self.B.data.normal_(mean=0.0, std=0.02)
        self.A.data.normal_(mean=0.0, std=0.02)
        self.lora = LinearLora(
            in_dim=in_dim,
            out_dim=out_dim,
            r=r,
            bias=True,
        )
        self.lora2 = LinearLora(
            in_dim=in_dim,
            out_dim=out_dim,
            r=r,
            bias=True,
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states @ (self.A @ self.B) + self.lora(hidden_states)
        return hidden_states



mse_loss = torch.nn.MSELoss(reduction="sum")


amount_steps = 10000


for layer_pos in range(max_layer):
    meta = model.model.layers[layer_pos].mlp.metadata
    lora_model = Lora(
        in_dim=config.hidden_size,
        out_dim=config.intermediate_size,
        r=64,
        bias=True,
    )
    lora_model.to(model.device)
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=0.001)

    for step in range(amount_steps):
        optimizer.zero_grad()

        lora_result = lora_model(meta["x"])
        labels = torch.vstack([meta["gate_proj(x)"]])

        predicts = torch.vstack([lora_result])
        loss = mse_loss(

            predicts,
            labels,
        )


        loss.backward()
        optimizer.step()
    print(f"Layer {layer_pos} Step: {step} - {loss.item()}")
```


```text
Layer 0 Step: 9999 - 0.03770807012915611
Layer 1 Step: 9999 - 0.43978697061538696
Layer 2 Step: 9999 - 17.232135772705078
Layer 3 Step: 9999 - 15.803369522094727
Layer 4 Step: 9999 - 40.39530944824219
Layer 5 Step: 9999 - 30.67352294921875
Layer 6 Step: 9999 - 0.8146815896034241
Layer 7 Step: 9999 - 101.2078857421875
Layer 8 Step: 9999 - 0.004654616117477417
Layer 9 Step: 9999 - 116.84696960449219
Layer 10 Step: 9999 - 133.5845947265625
Layer 11 Step: 9999 - 0.3355434238910675
Layer 12 Step: 9999 - 42.46553421020508
Layer 13 Step: 9999 - 0.21360622346401215
Layer 14 Step: 9999 - 93.36175537109375
Layer 15 Step: 9999 - 61.53105163574219
Layer 16 Step: 9999 - 179.4161376953125
Layer 17 Step: 9999 - 0.4249575734138489
Layer 18 Step: 9999 - 88.52590942382812
Layer 19 Step: 9999 - 0.3642119765281677
Layer 20 Step: 9999 - 0.1341780126094818
Layer 21 Step: 9999 - 140.767578125
Layer 22 Step: 9999 - 830.5631103515625
Layer 23 Step: 9999 - 0.03569779172539711
Layer 24 Step: 9999 - 4.814225196838379
Layer 25 Step: 9999 - 745.8604125976562
Layer 26 Step: 9999 - 22.504730224609375
Layer 27 Step: 9999 - 68.37686157226562
Layer 28 Step: 9999 - 776.0977783203125
Layer 29 Step: 9999 - 17.450828552246094
Layer 30 Step: 9999 - 870.1302490234375
Layer 31 Step: 9999 - 1.193930415865907e-06
```

Выводы:
- как минимум для одного конкретного примера мы можем подобрать низкоранговые веса, чтобы результат никак не изменился. 
- это ничего не значит, что-то как минимум то на чем можно делать выводы это когда я на 1000 примерах смогу получить схожий результат.
- я не знаю много это или мало для ошибки. чисто интуитивно там где 0 это приемлимый результат, а как на деле я не знаю.



### Experiment 2
Попробовал оптимизировать данную матрицу с рангом 1024. на 100 примерах из openorca.

```python
from torch.nn.parameter import Parameter


class LinearLora(torch.nn.Module):
    def __init__(self, in_dim=768, out_dim=768, r=16, bias=False):
        super().__init__()
        self.dense_h_to_4h = torch.nn.Linear(in_dim, r, bias=bias)
        self.dense_4h_to_h = torch.nn.Linear(r, out_dim, bias=bias)

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states




class Lora(torch.nn.Module):
    def __init__(
        self,
        in_dim=768,
        out_dim=768,
        r=8,
        bias=True,
    ):
        super().__init__()

        self.A = Parameter(torch.empty((in_dim, r)), requires_grad=True)
        self.B = Parameter(torch.empty((r, out_dim)), requires_grad=True)
        
        self.A1 = Parameter(torch.empty((in_dim, r)), requires_grad=True)
        self.B1 = Parameter(torch.empty((r, out_dim)), requires_grad=True)
        
        self.A.data.normal_(mean=0.0, std=0.02)
        self.B.data.normal_(mean=0.0, std=0.02)
        
        self.A1.data.normal_(mean=0.0, std=0.02)
        self.B1.data.normal_(mean=0.0, std=0.02)
        
        self.lora = LinearLora(
            in_dim=in_dim,
            out_dim=out_dim,
            r=r,
            bias=True,
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states @ (self.A @ self.B) + self.lora(hidden_states)
        # hidden_states = self.lora(hidden_states) + self.lora2(hidden_states)
        return hidden_states


mse_loss = torch.nn.MSELoss(reduction="sum")


amount_epochs = 10000

lora_model = Lora(
    in_dim=config.hidden_size,
    out_dim=config.intermediate_size,
    r=1024,
    bias=True,
)
lora_model.to(device)
lora_model.to(torch.float32)
optimizer = torch.optim.Adam(lora_model.parameters(), lr=0.00001)
for epoch in range(amount_epochs):
    train_total_loss = 0
    lora_model.train()
    for batch_id, batch in enumerate(train_tensor_dataset):
        optimizer.zero_grad()
        x = batch["x"].to(device)
        x = x.to(torch.float32)
        predicts = lora_model(x)
        labels = batch["gate_proj(x)"].to(device)
        labels = labels.to(torch.float32)

        loss = mse_loss(
            predicts,
            labels,
        )
        train_total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            lora_model.parameters(),
            1.0,
        )
        if (batch_id + 1) % 40:
            optimizer.step()
        # break
    # if epoch % 100 == 0:
    print(f"Epoch: {epoch} - train loss {train_total_loss / len(train_tensor_dataset)}")
    lora_model.eval()
    valid_total_loss = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(valid_tensor_dataset):
            x = batch["x"].to(device)
            x = x.to(torch.float32)
            predicts = lora_model(x)
            labels = batch["gate_proj(x)"].to(device)
            labels = labels.to(torch.float32)

            loss = mse_loss(
                predicts,
                labels,
            )
            valid_total_loss += loss.item()

        # break
    # if epoch % 100 == 0:
    print(f"Epoch: {epoch} - valid loss {valid_total_loss / len(valid_tensor_dataset)}")
    print("---")
    # break
# break
```
```text
Epoch: 0 - train loss 200171.3434375
Epoch: 0 - valid loss 143578.24013671876
---
Epoch: 1 - train loss 161983.2674121094
Epoch: 1 - valid loss 121752.9062109375
---
Epoch: 2 - train loss 140265.97559570312
Epoch: 2 - valid loss 107964.145703125
---
Epoch: 3 - train loss 124984.997578125
Epoch: 3 - valid loss 97572.98702148437
---
Epoch: 4 - train loss 112865.10587402344
Epoch: 4 - valid loss 89134.9857421875
---
Epoch: 5 - train loss 102821.86488769531
Epoch: 5 - valid loss 82074.70575195312
---
Epoch: 6 - train loss 94322.41747070312
Epoch: 6 - valid loss 76055.25157226562
---
Epoch: 7 - train loss 87013.919453125
Epoch: 7 - valid loss 70837.4691796875
---
Epoch: 8 - train loss 80636.41517089844
Epoch: 8 - valid loss 66248.62919433594
---
Epoch: 9 - train loss 75002.22755371094
Epoch: 9 - valid loss 62166.28244140625
---
Epoch: 10 - train loss 69976.69598876953
Epoch: 10 - valid loss 58501.59763671875
---
Epoch: 11 - train loss 65460.139921875
Epoch: 11 - valid loss 55187.63491943359
---
Epoch: 12 - train loss 61375.92646484375
Epoch: 12 - valid loss 52172.28962402344
---
Epoch: 13 - train loss 57663.29983642578
Epoch: 13 - valid loss 49414.030236816405
---
Epoch: 14 - train loss 54273.07332885742
Epoch: 14 - valid loss 46879.27219482422
---
Epoch: 15 - train loss 51164.954576416014
Epoch: 15 - valid loss 44540.641701660155
---
Epoch: 16 - train loss 48305.697044677734
Epoch: 16 - valid loss 42375.6615625
---
Epoch: 17 - train loss 45667.65075317383
Epoch: 17 - valid loss 40365.69401367188
---
Epoch: 18 - train loss 43227.58079345703
Epoch: 18 - valid loss 38495.07844482422
---
Epoch: 19 - train loss 40965.71959594727
Epoch: 19 - valid loss 36750.46581298828
---
Epoch: 20 - train loss 38865.03975097656
Epoch: 20 - valid loss 35120.31932006836
---
Epoch: 21 - train loss 36910.71265563965
Epoch: 21 - valid loss 33594.561224365236
---
Epoch: 22 - train loss 35089.71127990723
Epoch: 22 - valid loss 32164.310434570314
---
Epoch: 23 - train loss 33390.51142822266
Epoch: 23 - valid loss 30821.69111694336
---
Epoch: 24 - train loss 31802.863746948242
Epoch: 24 - valid loss 29559.683031005858
---
Epoch: 25 - train loss 30317.615788574218
Epoch: 25 - valid loss 28372.00634277344
---
Epoch: 26 - train loss 28926.562989501952
Epoch: 26 - valid loss 27253.022591552734
---
Epoch: 27 - train loss 27622.329516601563
Epoch: 27 - valid loss 26197.65035461426
---
Epoch: 28 - train loss 26398.263056640626
Epoch: 28 - valid loss 25201.297755126954
---
Epoch: 29 - train loss 25248.35076904297
Epoch: 29 - valid loss 24259.800334472657
---
Epoch: 30 - train loss 24167.138350524903
Epoch: 30 - valid loss 23369.370201416015
---
Epoch: 31 - train loss 23149.665183410645
Epoch: 31 - valid loss 22526.549543457033
---
Epoch: 32 - train loss 22191.40869140625
Epoch: 32 - valid loss 21728.17476135254
---
Epoch: 33 - train loss 21288.23466430664
Epoch: 33 - valid loss 20971.34056640625
---
Epoch: 34 - train loss 20436.35097076416
Epoch: 34 - valid loss 20253.373647460936
---
Epoch: 35 - train loss 19632.273422546386
Epoch: 35 - valid loss 19571.807697753906
---
Epoch: 36 - train loss 18872.792532348634
Epoch: 36 - valid loss 18924.361422729493
---
Epoch: 37 - train loss 18154.947302856446
Epoch: 37 - valid loss 18308.921630249024
---
Epoch: 38 - train loss 17476.000011901855
Epoch: 38 - valid loss 17723.527742004393
---
Epoch: 39 - train loss 16833.41700073242
Epoch: 39 - valid loss 17166.35824951172
---
Epoch: 40 - train loss 16224.85169128418
Epoch: 40 - valid loss 16635.71823730469
---
Epoch: 41 - train loss 15648.12684173584
Epoch: 41 - valid loss 16130.030386962891
---
Epoch: 42 - train loss 15101.223155517579
Epoch: 42 - valid loss 15647.825400085449
---
Epoch: 43 - train loss 14582.265806121826
Epoch: 43 - valid loss 15187.73372680664
---
Epoch: 44 - train loss 14089.514256591798
Epoch: 44 - valid loss 14748.478555297852
---
Epoch: 45 - train loss 13621.350973815917
Epoch: 45 - valid loss 14328.86874267578
---
Epoch: 46 - train loss 13176.271868286132
Epoch: 46 - valid loss 13927.792131652832
---
Epoch: 47 - train loss 12752.879300384522
Epoch: 47 - valid loss 13544.210612487794
---
Epoch: 48 - train loss 12349.872951049805
Epoch: 48 - valid loss 13177.154028930665
---
Epoch: 49 - train loss 11966.041887512207
Epoch: 49 - valid loss 12825.715635375976
---
Epoch: 50 - train loss 11600.258267059326
Epoch: 50 - valid loss 12489.046991882324
---
Epoch: 51 - train loss 11251.471334228516
Epoch: 51 - valid loss 12166.35433227539
---
Epoch: 52 - train loss 10918.700881347657
Epoch: 52 - valid loss 11856.893110351562
---
Epoch: 53 - train loss 10601.031786499023
Epoch: 53 - valid loss 11559.96663909912
---
Epoch: 54 - train loss 10297.60971359253
Epoch: 54 - valid loss 11274.920405578614
---
Epoch: 55 - train loss 10007.636294555665
Epoch: 55 - valid loss 11001.140187072753
---
Epoch: 56 - train loss 9730.364652633667
Epoch: 56 - valid loss 10738.04932647705
---
Epoch: 57 - train loss 9465.095657958984
Epoch: 57 - valid loss 10485.105767669678
---
Epoch: 58 - train loss 9211.17499320984
Epoch: 58 - valid loss 10241.799250946046
---
Epoch: 59 - train loss 8967.98931930542
Epoch: 59 - valid loss 10007.649899291991
---
Epoch: 60 - train loss 8734.963890838622
Epoch: 60 - valid loss 9782.205735473633
---
Epoch: 61 - train loss 8511.559084625243
Epoch: 61 - valid loss 9565.040718231201
---
Epoch: 62 - train loss 8297.268273849488
Epoch: 62 - valid loss 9355.75320022583
---
Epoch: 63 - train loss 8091.61596786499
Epoch: 63 - valid loss 9153.96499572754
---
Epoch: 64 - train loss 7894.155486221313
Epoch: 64 - valid loss 8959.318712158203
---
Epoch: 65 - train loss 7704.467024612427
Epoch: 65 - valid loss 8771.477104187012
---
Epoch: 66 - train loss 7522.1552102661135
Epoch: 66 - valid loss 8590.122151184081
---
Epoch: 67 - train loss 7346.848664321899
Epoch: 67 - valid loss 8414.953182067871
---
Epoch: 68 - train loss 7178.197667922974
Epoch: 68 - valid loss 8245.686357269287
---
Epoch: 69 - train loss 7015.872951660156
Epoch: 69 - valid loss 8082.053329925537
---
Epoch: 70 - train loss 6859.564350357055
Epoch: 70 - valid loss 7923.800392761231
---
Epoch: 71 - train loss 6708.979876403809
Epoch: 71 - valid loss 7770.687978668213
---
Epoch: 72 - train loss 6563.84421295166
Epoch: 72 - valid loss 7622.4890989685055
---
Epoch: 73 - train loss 6423.897626342773
Epoch: 73 - valid loss 7478.989275360107
---
Epoch: 74 - train loss 6288.895475082398
Epoch: 74 - valid loss 7339.985270690918
---
Epoch: 75 - train loss 6158.606517372132
Epoch: 75 - valid loss 7205.284507751465
---
Epoch: 76 - train loss 6032.813006858825
Epoch: 76 - valid loss 7074.705162963867
---
Epoch: 77 - train loss 5911.308802261353
Epoch: 77 - valid loss 6948.074246520996
---
Epoch: 78 - train loss 5793.89946773529
Epoch: 78 - valid loss 6825.2279596710205
---
Epoch: 79 - train loss 5680.401057357788
Epoch: 79 - valid loss 6706.010909805298
---
Epoch: 80 - train loss 5570.639730072022
Epoch: 80 - valid loss 6590.27577041626
---
Epoch: 81 - train loss 5464.451122398376
Epoch: 81 - valid loss 6477.882376403809
---
...
Epoch: 3459 - valid loss 250.0959122610092
---
Epoch: 3460 - train loss 168.44642386078834
Epoch: 3460 - valid loss 249.98123679637908
---
Epoch: 3461 - train loss 168.5229579257965
Epoch: 3461 - valid loss 250.08856869220733
---
Epoch: 3462 - train loss 168.48945083856583
Epoch: 3462 - valid loss 250.03873988628388
---
Epoch: 3463 - train loss 168.48320989131926
Epoch: 3463 - valid loss 250.199831905365
---
Epoch: 3464 - train loss 168.51102076172828
Epoch: 3464 - valid loss 250.37397064685823
---
Epoch: 3465 - train loss 168.4909599816799
Epoch: 3465 - valid loss 250.1177248120308
---
Epoch: 3466 - train loss 168.6001497757435
Epoch: 3466 - valid loss 250.37656612634657
---
Epoch: 3467 - train loss 168.5197152709961
Epoch: 3467 - valid loss 250.1486878681183
---
Epoch: 3468 - train loss 168.3810770201683
Epoch: 3468 - valid loss 250.01372981786727
---
Epoch: 3469 - train loss 168.33698074460028
Epoch: 3469 - valid loss 250.1920231485367
---
Epoch: 3470 - train loss 168.3395501089096
Epoch: 3470 - valid loss 250.3504771256447
---
Epoch: 3471 - train loss 168.36196244478225
Epoch: 3471 - valid loss 250.1830565595627
---
Epoch: 3472 - train loss 168.51347168922425
Epoch: 3472 - valid loss 250.0970231413841
---
Epoch: 3473 - train loss 168.45227784395217
Epoch: 3473 - valid loss 250.04897697448732
---
Epoch: 3474 - train loss 168.32285446882247
Epoch: 3474 - valid loss 249.95669543266297
---
Epoch: 3475 - train loss 168.31174188375473
Epoch: 3475 - valid loss 250.22072457313539
---
Epoch: 3476 - train loss 168.24036512613296
Epoch: 3476 - valid loss 250.01338220119476
---
Epoch: 3477 - train loss 168.19590500593185
Epoch: 3477 - valid loss 250.3404205107689
---
Epoch: 3478 - train loss 168.2045400631428
Epoch: 3478 - valid loss 250.185750439167
---
Epoch: 3479 - train loss 168.26338461518287
Epoch: 3479 - valid loss 250.03093426942826
---
Epoch: 3480 - train loss 168.24487315654756
Epoch: 3480 - valid loss 250.25296383857727
---
Epoch: 3481 - train loss 168.27222452044487
Epoch: 3481 - valid loss 250.13988070487977
---
Epoch: 3482 - train loss 168.12705314159393
Epoch: 3482 - valid loss 250.06844162464142
---
Epoch: 3483 - train loss 168.15406739354134
Epoch: 3483 - valid loss 250.07136808872224
---
Epoch: 3484 - train loss 168.11912911891937
Epoch: 3484 - valid loss 250.1176122713089
---
Epoch: 3485 - train loss 168.1148617863655
Epoch: 3485 - valid loss 250.11060014009476
---
Epoch: 3486 - train loss 168.1000016772747
Epoch: 3486 - valid loss 250.12191313028336
---
Epoch: 3487 - train loss 168.15239727258682
Epoch: 3487 - valid loss 250.1320447707176
---
Epoch: 3488 - train loss 168.14642921209335
Epoch: 3488 - valid loss 249.96136803150176
---
Epoch: 3489 - train loss 168.09610471248627
Epoch: 3489 - valid loss 249.93430508136748
---
Epoch: 3490 - train loss 168.09288675546645
Epoch: 3490 - valid loss 250.0543128991127
---
Epoch: 3491 - train loss 168.11840978622436
Epoch: 3491 - valid loss 250.06626747608186
...
Epoch: 4624 - train loss 165.70102674603461
Epoch: 4624 - valid loss 246.57105510234834
---
Epoch: 4625 - train loss 165.6300622713566
Epoch: 4625 - valid loss 246.59646065950395
---
Epoch: 4626 - train loss 165.61519354343415
Epoch: 4626 - valid loss 246.52468415737152
---
Epoch: 4627 - train loss 165.7316993510723
Epoch: 4627 - valid loss 246.61669226408006
---
Epoch: 4628 - train loss 165.706702439785
Epoch: 4628 - valid loss 246.38423814296723
---
Epoch: 4629 - train loss 165.7604597878456
---
```

Пробовал до этого оптимизировать батчами, но тогда получается что в памяти нужно хранить матрицы 4096х4096 на каждый пример, это больше 250GB для 100 примеров. Плюс для батч получается очень маленький для такого рода задачи, занимает почти 29GB для batch_size=16.

К тому же скорее всего из-за большого количества векторов паддингов модель плохо обобщалась.

```text
Epoch: 0 - 118209748.57142857
Epoch: 100 - 1461227.4955357143
Epoch: 200 - 1005819.9441964285
Epoch: 300 - 629241.3504464285
Epoch: 400 - 511065.4754464286
Epoch: 500 - 162908.36886160713
Epoch: 600 - 157512.14732142858
Epoch: 700 - 237796.56361607142
Epoch: 800 - 119738.515625
Epoch: 900 - 75528.10770089286
Epoch: 1000 - 267220172.57142857
Epoch: 1100 - 103504.44810267857
Epoch: 1200 - 86821.23828125
Epoch: 1300 - 82264.361328125
Epoch: 1400 - 418591.8917410714
Epoch: 1500 - 87370.48772321429
Epoch: 1600 - 112994.123046875
Epoch: 1700 - 142490.16629464287
Epoch: 1800 - 82327.04938616071
Epoch: 1900 - 78044.42745535714
Epoch: 2000 - 78206.6953125
Epoch: 2100 - 89908.03738839286
Epoch: 2200 - 79061.66964285714
Epoch: 2300 - 65599.01618303571
Epoch: 2400 - 58484.57366071428
Epoch: 2500 - 92165.03069196429
Epoch: 2600 - 74148.73521205357
Epoch: 2700 - 66937.294921875
Epoch: 2800 - 62927.44419642857
Epoch: 2900 - 187429.76088169642
Epoch: 3000 - 129819.40457589286
Epoch: 3100 - 56979.361607142855
Epoch: 3200 - 57186.49079241072
Epoch: 3300 - 56911.13950892857
Epoch: 3400 - 54044.484095982145
Epoch: 3500 - 55826.873325892855
Epoch: 3600 - 56904.90178571428
Epoch: 3700 - 54797.65625
Epoch: 3800 - 55369.04213169643
Epoch: 3900 - 55433.94056919643
Epoch: 4000 - 58600.556919642855
Epoch: 4100 - 133469.42745535713
Epoch: 4200 - 79545.958984375
Epoch: 4300 - 67670.79380580357
Epoch: 4400 - 61266.314174107145
Epoch: 4500 - 57551.37555803572
Epoch: 4600 - 54915.49358258928
Epoch: 4700 - 52943.14508928572
Epoch: 4800 - 51684.585658482145
Epoch: 4900 - 50287.44252232143
Epoch: 5000 - 49451.91127232143
Epoch: 5100 - 53936.556640625
Epoch: 5200 - 50148.244419642855
Epoch: 5300 - 47788.056361607145
Epoch: 5400 - 48099.45228794643
Epoch: 5500 - 48028.46763392857
Epoch: 5600 - 51862.701450892855
Epoch: 5700 - 48411.86830357143
Epoch: 5800 - 51840.85463169643
Epoch: 5900 - 153519.06082589287
Epoch: 6000 - 94530.02008928571
Epoch: 6100 - 76068.78404017857
Epoch: 6200 - 66085.99386160714
Epoch: 6300 - 59883.17103794643
Epoch: 6400 - 55829.53989955357
Epoch: 6500 - 52778.66908482143
Epoch: 6600 - 76904.16183035714
Epoch: 6700 - 49362.25111607143
Epoch: 6800 - 48413.625
Epoch: 6900 - 47719.078404017855
Epoch: 7000 - 51701.80608258928
Epoch: 7100 - 47922.55189732143
Epoch: 7200 - 48086.69029017857
Epoch: 7300 - 46664.076450892855
Epoch: 7400 - 49945.26841517857
Epoch: 7500 - 49286.161830357145
Epoch: 7600 - 48500.679966517855
Epoch: 7700 - 149010.5279017857
``` 

### Experiment 3

Использовал стандартное сингулярное разложение для того чтобы получить низкоранговую матрицу. размерность равна 16. Применил лору только к 17 слою.

Модель Open-Orca/Mistral-7B-OpenOrca.
- оригинальные параметры 7241748480
- после применения lora 7066472448

Лог эвалюации на https://huggingface.co/datasets/openaccess-ai-collective/oo-gpt4-filtered.
```text
1000
teacher_loss 2.297937617301941 tensor(9.9536) batch = 2
student_loss 2.2953803930282595 tensor(9.9282) batch = 2
10000
teacher_loss 2.3349632108330725 tensor(10.3291) batch = 2
student_loss 2.3332279333114623 tensor(10.3112) batch = 2
100000
teacher_loss 2.3350424207425116 tensor(10.3299) batch = 2
student_loss 2.3333492525792123 tensor(10.3124) batch = 2
```

После сокращения размерности модель даже улучшила качество.
- [commit](https://github.com/dmitrymailk/gpus_go_brrrr/blob/44667834f2107167f048fedfc27d0bec1534d298/ebany_research/llm_lora/original_svd.py#L248)

### Experiment 4

Сделал свой сплит данного датасета, чтобы можно было честно сравнивать модели после каллибровки. Соответственно все прошлые метрики не считаются.

