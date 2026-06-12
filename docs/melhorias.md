# Melhorias — Bloco 2: lacunas científicas para a 5ª entrega

**Status:** ✅ Implementado e commitado em `main` (12/06/2026); execução de GPU do item 2.1 pendente.
**Escopo:** quatro itens do checklist Pré-Entrega 5ª que faltavam após o MVP (S1–S5) — ablation do λ, estratificação por tamanho de lesão, análise de falha estruturada e orquestrador de figuras.
**Natureza:** três itens (2.2, 2.3, 2.4) rodam pós-treino (CPU) sobre artefatos já existentes; um item (2.1) exige treino em GPU.

---

## 1. Objetivo

Fechar as lacunas entre o MVP defensável da 4ª entrega e o TCC completo da 5ª
entrega, sem alterar nenhum resultado já consolidado no S5. Cada item foi
escrito reaproveitando a infraestrutura existente (schema do CSV por slice do
S5.8, calibração CQR/ResM, helpers estatísticos do S5.9) para garantir que a
análise nova seja consistente com a já publicada.

Os achados centrais do S5 permanecem o pano de fundo: a vantagem de cobertura
do Grupo A é um efeito de **calibração** (escala localmente adaptativa), não de
arquitetura; a sub-cobertura em lesão é um problema **intra-sequência**; o
efeito da loss QR-Lesion é pequeno mas confiável (d_z ~0.19–0.26) e o de
calibração é grande (d_z ~0.93).

## 2. Visão geral

| Item | Script | Custo | Commit |
| --- | --- | --- | --- |
| 2.1 Ablation do λ | `scripts/ablation_lambda.py` | GPU (3 treinos) | `0ca0ed4` |
| 2.2 Estratificação por tamanho | `scripts/stratified_analysis.py` | CPU | `ccd3f13` |
| 2.3 Análise de falha | `scripts/failure_analysis.py` | CPU | `890d57a` |
| 2.4 Hero figure + orquestrador | `scripts/plot_figures.py` | CPU | `b211bc9` |

Sequência adotada: CPU primeiro (2.2/2.3/2.4 sobre os artefatos do S5), GPU por
último (2.1 consome a quota de treino). Todos os scripts têm cabeçalho
Autor/Data/Descrição, sem nomes ou dados sensíveis no corpo, validados por
`py_compile` e por teste funcional com dados sintéticos antes do commit.

## 3. Item 2.1 — Curva de ablation do λ

### 3.1 Problema

O experimento fixou λ=5 sem a curva que justifica essa escolha. Sem ela, não há
como mostrar que λ=5 é o melhor trade-off e não um valor arbitrário — e ablation
é precisamente o que separa "escolhi 5" de "5 é o ponto ótimo".

### 3.2 Decisões

**Conjunto de λ.** A curva final cobre λ ∈ {1, 3, 5, 10, 15}. Apenas
{3, 10, 15} são treinados; os demais são âncoras de custo zero (ver 3.3).

**Âncoras de custo zero.** A loss ponderada usa `w = 1 + (λ − 1)·M`, com M a
máscara binária de lesão. Logo:

- **λ = 1** ⇒ `w = 1` em todo pixel ⇒ a loss reduz exatamente ao `qr_loss` do
  **Grupo B**. O ponto λ=1 sai do `metrics_B.csv` já existente.
- **λ = 5** ⇒ é o **Grupo C** já treinado. O ponto sai do `metrics_C.csv`.

Isso reduz a curva de 5 pontos para apenas **3 treinos novos**.

**Iterações fixas em 210k.** Todos os λ usam os mesmos 210k iters das âncoras.
Reduzir as iterações dos λ novos confundiria *efeito do λ* com *tempo de treino*
— o que invalidaria a curva como argumento de trade-off. A seed fixa (42) entre
runs isola o efeito do λ da inicialização (Demšar, 2006, §3.2).

**Métricas plotadas.** ULAS_lesion, Width_lesion e Coverage_lesion vs λ, com IC
95% por bootstrap BCa (Efron & Tibshirani, 1993). Coverage tem linha de
referência nominal (1 − α = 0.90).

### 3.3 Reprodução

Treino (Kaggle T4, ~12 GPU-h por λ novo, ~36 GPU-h no total; idempotente e
resumível — pula etapa cuja saída já existe e o treino retoma de `last.pt`):

```
python scripts/ablation_lambda.py --lambdas 3 10 15 --total-iters 210000 \
  --recons-dir <recons> --masks-dir <masks> --work-dir <work> \
  --metrics-b <metrics_B.csv> --metrics-c <metrics_C.csv>
```

Montagem da curva e figuras a partir dos CSVs (CPU, sem treino):

```
python scripts/ablation_lambda.py --no-train --work-dir <work> \
  --metrics-b <metrics_B.csv> --metrics-c <metrics_C.csv> --lambdas 3 10 15
```

Saídas: `ablation_lambda.csv` e `figures/ablation_lambda_{ulas,width,coverage}_lesion.png` (300 DPI).

## 4. Item 2.2 — Estratificação por tamanho de lesão

### 4.1 Decisões

**Thresholds.** Mantidos os valores data-driven já commitados em
`configs/lesion_thresholds.json`: pequena <200, média 200–2000, grande >2000
(px²), definidos apenas no train para evitar leakage (Demšar, 2006). O script lê
os thresholds do JSON — alterar as faixas é editar a config, não o código.

**Atribuição de slice à faixa.** Cada slice é binado pela **área da sua máscara
de lesão** (`n_pixels_lesion`), que é o suporte exato onde Coverage_lesion,
Width_lesion e ULAS_lesion são computados. Isso evita o descasamento
bbox-vs-máscara em slices com múltiplas lesões e torna a estratificação
internamente consistente com as métricas. Como a máscara independe do grupo, o
n por faixa é idêntico em A/B/C e é reportado uma única vez.

**Incerteza estatística.** IC 95% por bootstrap BCa, com flag de poder por faixa
(n≥30 sem flag; 10≤n<30 marcado com `*`; n<10 com `**`), seguindo o aviso do
guia sobre n pequeno. O seed do bootstrap é determinístico por faixa×métrica
(evita variação por `PYTHONHASHSEED`), para a banca reproduzir os ICs.

### 4.2 Ressalva metodológica

Com 47 volumes de test, a faixa **grande (>2000 px²)** tende a cair em n<30
(possivelmente n<10) — o script expõe isso via o n por faixa e as flags, e a
escrita deve interpretar essa faixa com cautela. Permanece o confound
**sequência × tamanho** (a maioria das lesões pequenas concentra-se em AXFLAIR),
que limita a leitura causal da estratificação por tamanho isolada.

### 4.3 Reprodução

```
python scripts/stratified_analysis.py --csv-dir <dir com metrics_A/B/C.csv>
```

Saídas: `results/stratified_by_size.csv` (tidy), `docs/figures/stratified_by_size.md`
(tabela por métrica) e `figures/coverage_by_lesion_size.png` (300 DPI).

## 5. Item 2.3 — Análise de falha estruturada

### 5.1 Decisões

**Critério de pior caso.** Default = gap de cobertura em lesão,
`(1 − α) − Coverage_lesion`. A garantia do CQR é marginal, não condicional por
subgrupo (Romano et al., 2019); a sub-cobertura concentrada em lesão é a falha
que o método não promete cobrir, e ranquear pelo gap a expõe diretamente.
Alternativa `--rank-by ulas` ordena por ULAS_lesion ascendente.

**Pipeline em duas etapas.** Ranqueia do `metrics_C.csv` (sem GPU) e só faz
forward nos K piores para renderizar — custo de GPU desprezível.

**Atributo de artefato.** A coluna `artifact_type` do CSV é deixada em branco de
propósito: tipo de artefato não é inferível do CSV e é anotação manual do
especialista. Não foi inventado um classificador automático.

**Figura.** Painel GT | recon | |erro| | largura do intervalo | falha de
cobertura (pixels de lesão fora do intervalo em vermelho), com contorno da lesão,
300 DPI.

### 5.2 Reprodução

```
python scripts/failure_analysis.py --metrics-csv <metrics_C.csv> --group C \
  --checkpoint <best.pt> --qhat <q_hat_C.json> \
  --recons-dir <recons> --masks-dir <masks> \
  --n-worst 10 --n-figures 3 --rank-by coverage_gap
```

Saídas: `results/failure_top10_groupC.csv` e `figures/failures/*.png`. Sem
`--checkpoint`/`--qhat`, gera só o CSV (ranqueamento).

## 6. Item 2.4 — Hero figure e orquestrador de figuras

### 6.1 Decisões

**Incerteza plotada.** Halfwidth pós-calibração `(upper_cal − lower_cal)/2`, a
mesma definição do `notebooks/demo.ipynb`, para a figura do script e a do
notebook coincidirem.

**Escala compartilhada.** As três colunas de incerteza (A/B/C) usam a mesma
escala de cor (vmax = 99º percentil sobre os três mapas), para comparação visual
honesta entre grupos (Tufte, 2001).

**Bbox por componente.** Bounding box por componente conexo de lesão via
`scipy.ndimage.label`, com fallback para bbox único.

**Seleção de slice.** Default = slice de maior área de lesão no test (mais
ilustrativo); override por `--volume`/`--slice`.

**Orquestração.** `--which {hero,stratified,failures,histogram,all}` regenera as
figuras por um comando, chamando os scripts já commitados; `all` faz o que for
possível e apenas avisa (não aborta) quando faltam checkpoints.

### 6.2 Reprodução

```
python scripts/plot_figures.py --which hero \
  --ckpt-a <A/best.pt> --ckpt-b <B/best.pt> --ckpt-c <C/best.pt> \
  --qhat-a <q_hat_A.json> --qhat-b <q_hat_B.json> --qhat-c <q_hat_C.json> \
  --recons-dir <recons> --masks-dir <masks>
```

Saída: `figures/hero_figure.png` (300 DPI).

## 7. Referências

- Romano, Y.; Patterson, E.; Candès, E. (2019). Conformalized Quantile Regression. NeurIPS 32, 3543–3553.
- Efron, B.; Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman & Hall (BCa).
- Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR 7:1–30.
- Angelopoulos, A. N.; Bates, S. (2023). Conformal Prediction: A Gentle Introduction. Foundations and Trends in ML 16(4):494–591.
- Tufte, E. R. (2001). The Visual Display of Quantitative Information. 2ª ed., Graphics Press.
