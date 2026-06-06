# S5-extras — Análises post-hoc de calibração e isolamento de mecanismo

**Trabalho de Conclusão de Curso**
**Programa**: USP-ESALQ / PECEGE — Data Science & Analytics
**Versão**: extensão da 4ª entrega (análises post-hoc, sem re-treino)
**Repositório**: github.com/KR0N0S7/tcc-mri-uncertainty
**Branch**: feat/S5-extras
**Data deste relatório**: 04/06/2026

---

## Resumo executivo

Este documento estende a 4ª entrega (S5) com cinco análises post-hoc executadas sobre os mesmos checkpoints (Grupos A/B/C) e os mesmos `metrics_*.csv` por slice do S5.8 — nenhuma exige re-treino. O objetivo é responder, com rigor, à pergunta deixada em aberto pelo S5: a superioridade do Grupo A (ResM) em cobertura de lesão (~8 pontos percentuais sobre o CQR) vem da **arquitetura** ou do **esquema de calibração** (multiplicativo *locally-adaptive* vs aditivo marginal)?

As cinco análises são: (1) um quarto calibrador — **CQR normalizado** (localmente ponderado, Lei et al. 2018, Sec. 5.2) aplicado aos modelos B/C sem alterar a arquitetura, para isolar o mecanismo; (2) **fronteira de eficiência** (cobertura a largura igualada) para comparação justa entre métodos; (3) **gap de cobertura condicional** (global vs lesão, por sequência, por carga lesional) com intervalos Clopper-Pearson; (4) **curva de confiabilidade** (cobertura empírica vs nominal) em uma grade de níveis; (5) **tamanho de efeito e IC sobre o delta pareado** (BCa, rank-biserial, d_z de Cohen), corrigindo o objeto estatístico do S5.9. Em complemento, a Seção 4.6 testa a calibração Mondrian por sequência.

**Achados principais:**

1. **A vantagem do ResM é mecanismo de calibração, não arquitetura — confirmado.** Trocando o CQR de aditivo (q constante) para normalizado (escala multiplicativa pela largura predita `w(x) = upper(x) - lower(x)`), nos mesmos modelos B/C, a cobertura em lesão (nível nominal 0.90, micro-média) sobe de 0,728/0,737 para 0,848/0,860 — fecha e ultrapassa a do ResM (0,816) — e melhora a calibração global (0,876 para 0,894). A adaptatividade local da calibração é, portanto, o fator causal dos ~8 pp observados no S5.

2. **A forma da adaptatividade importa: o ResM é o mais eficiente.** A fronteira de eficiência (cobertura a largura igualada) ordena os métodos como ResM superior ao CQR aditivo, que por sua vez supera o CQR normalizado. O CQR normalizado só atinge cobertura alta às custas de intervalos mais largos em lesão (0,072–0,078 vs 0,055 do ResM). A escala local aprendida do ResM, `u(x)` (treinada para prever a magnitude do resíduo), é uma variável de condicionamento mais eficiente, em lesão, que a largura quantílica `w(x)`.

3. **A sub-cobertura em lesão é concentrada em T1 e cresce com a carga lesional.** O gap condicional (global menos lesão, micro) é de ~7,6 pp para o ResM e ~14–15 pp para o CQR aditivo. Estratificando, a falha é dominada pela sequência AXT1 (cobertura em lesão 0,61–0,73), enquanto AXT1POST é bem coberta (~0,87). A cobertura também degrada monotonicamente com a carga lesional do slice.

4. **A magnitude do efeito da loss QR-Lesion (C vs B) é pequena; a da calibração é grande.** No delta pareado por slice, C supera B em Coverage_lesion, Width_lesion e ULAS_lesion com IC 95% BCa que exclui zero, porém com tamanho de efeito pequeno (d_z 0,19–0,26). Já a diferença ResM vs CQR em Coverage_lesion é grande (d_z 0,93). H4 (IoU_topk_lesion) permanece indistinguível de zero (IC inclui zero), consistente com o S5.9.

5. **ULAS diverge da cobertura — evidência de validade de construto.** Em ULAS_lesion, os métodos CQR (B/C) superam o ResM (delta pareado A menos B = -0,021, d_z -0,77), apesar de o ResM cobrir mais. Cobertura e ULAS medem coisas diferentes: o ResM ganha em cobertura, o CQR ganha em alinhamento direcional. Isso reforça que o ULAS não é uma reformulação da cobertura.

6. **Mondrian por sequência não conserta a lesão.** Calibrar `q` por sequência corrige a má-calibração global entre sequências do CQR aditivo, mas não reduz — e em AXT1 piora — a sub-cobertura em lesão (B: 0,612 → 0,579), porque o `q` por sequência é dominado pelos pixels de fundo. Para ResM e CQR normalizado é um no-op (a adaptatividade local já homogeneíza as sequências). Conclui-se que a sub-cobertura em lesão é intra-sequência (lesão vs fundo); a alavanca é a adaptatividade local ou um Mondrian por estrato lesão/fundo.

**Interpretação geral.** A narrativa do trabalho fica mais precisa: em quantificação de incerteza pixelwise para regiões de lesão, o fator dominante da cobertura condicional é a adaptatividade local da calibração (não a arquitetura nem a loss); entre escalas locais, a magnitude de resíduo aprendida supera a largura quantílica em eficiência; e a sub-cobertura residual é dependente de sequência. A contribuição original (loss QR-Lesion, Grupo C) permanece válida com efeito pequeno e confiável, e o ULAS ganha validação por divergir da cobertura.

---

## 1. Contexto e motivação

O S5 (4ª entrega) reportou dois achados centrais: (i) a loss QR-Lesion (Grupo C) supera o baseline QR (Grupo B) em métricas de lesão com efeito pequeno mas estatisticamente robusto; (ii) inesperadamente, o ResM (Grupo A) supera ambos em Coverage_lesion por ~8 pontos percentuais. O S5 atribuiu (ii) à adaptatividade local do intervalo `[x - q*u(x), x + q*u(x)]`, mas não isolou a causa — a diferença entre A e B/C confunde arquitetura (ResM vs QR) e esquema de calibração (multiplicativo vs aditivo). Estas análises foram desenhadas para desconfundir esse achado e para tornar as comparações justas e estatisticamente honestas, sem custo de re-treino.

---

## 2. Métodos

### 2.1 Item 1 — CQR normalizado (locally-weighted)

Aplica o resíduo localmente ponderado de Lei et al. (2018, Sec. 5.2) ao score CQR de Romano et al. (2019). Score aditivo (B/C original): `E(x,y) = max(lower(x) - y, y - upper(x))`, intervalo `[lower - q, upper + q]`. Score normalizado: `E~(x,y) = E(x,y) / w(x)` com `w(x) = max(upper(x) - lower(x), eps)`, intervalo `[lower - q*w(x), upper + q*w(x)]`. Como `w(x) > 0`, vale a equivalência `y in intervalo <=> E~ <= q`, e como `w(x)` é função apenas de x (saída do modelo ajustada no treino), o score normalizado permanece exchangeable cal-test — a garantia de cobertura marginal pixelwise é preservada (Lei et al. 2018; Vovk et al. 2005). Implementação em `src/calibration/adaptive_cqr.py`. Roda sobre os checkpoints B/C existentes (sem re-treino).

### 2.2 Itens 2 e 4 — varredura de nível nominal

Para cada calibrador (`scaled` = A; `cqr` = B/C aditivo; `cqr_norm` = B/C normalizado), calibra-se `q` sobre o split cal em uma grade de 15 níveis nominais (0,50 a 0,99) e avalia-se no split test a cobertura empírica e a largura média, em global e em lesão. Como cobertura(q) = fração de pixels com score <= q e a largura é afim em q, uma passada cal + uma passada test cobrem toda a grade. A fronteira de eficiência (item 2) é o plano largura-em-lesão vs cobertura-em-lesão; a curva de confiabilidade (item 4) é nominal vs empírico. Implementação em `scripts/analyze_calibration_sweep.py` e `scripts/plot_calibration_extras.py`.

### 2.3 Item 3 — cobertura condicional

Sobre os `metrics_*.csv` do S5.8, agrega cobertura como proporção binomial exata (Clopper-Pearson 1934) por estrato: lesão vs global; por sequência (AXFLAIR/AXT1/AXT1POST); por tercil de carga lesional do slice. Cobertura condicional exata é impossível em distribution-free (Barber et al. 2021); estes estratos discretos finitos são a versão atingível (particionamento estilo Mondrian, Vovk et al. 2005). Implementação em `scripts/analyze_conditional_coverage.py`.

### 2.4 Item 5 — delta pareado

Alinha os `metrics_*.csv` por (volume_id, slice_idx) e, por métrica e par (A-B, A-C, B-C), reporta a média do delta pareado, IC 95% BCa do delta (Efron & Tibshirani 1993), correlação rank-biserial pareada (tamanho de efeito do Wilcoxon, Kerby 2014) e d_z de Cohen pareado. Corrige o objeto estatístico do S5.9, que aplicou BCa sobre médias por grupo em vez de sobre o delta pareado. Implementação em `scripts/analyze_paired_deltas.py`.

### 2.5 Mondrian por sequência

Calibra um `q` por sequência (AXFLAIR/AXT1/AXT1POST) sobre o split cal e avalia no test, comparando ao esquema marginal (q único). Mondrian conformal (Vovk et al. 2005) entrega cobertura condicional ao estrato discreto. Implementação em `src/calibration/mondrian.py` e `scripts/analyze_mondrian_coverage.py`.

---

## 3. Validação do pipeline

Os resultados reais executados no Kaggle confirmaram a integridade do pipeline: os `q_hat` recomputados na varredura reproduzem exatamente os da calibração S5.7 — A (scaled) 2,052745, B (cqr) 0,010275, C (cqr) 0,010269 — para todas as casas decimais reportadas. Os SHA-256 dos checkpoints usados (Seção 8.1) coincidem com os do S5.7/S5.8. A varredura processou 730 slices de calibração (70.759.312 pixels) e 736 slices de test (68.366.624 pixels globais, 1.981.221 pixels de lesão), os mesmos totais do S5.8. Portanto, as diferenças numéricas entre este documento e o S5.8 decorrem exclusivamente da definição de agregação (micro vs macro; ver Seção 5.1), não de divergência de pipeline.

---

## 4. Resultados

### 4.1 Item 1 — isolamento de mecanismo (nível nominal 0.90, micro-média)

| Série | calibrador | q_hat | Coverage_global | Coverage_lesion | Width_global | Width_lesion |
|---|---|---|---|---|---|---|
| A · ResM | scaled | 2,0527 | 0,8922 | **0,8164** | 0,0310 | 0,0555 |
| B · QR | cqr | 0,0103 | 0,8763 | 0,7277 | 0,0356 | 0,0475 |
| B · QR | **cqr_norm** | 0,8435 | 0,8941 | **0,8481** | 0,0404 | 0,0725 |
| C · QR-Lesion | cqr | 0,0103 | 0,8771 | 0,7371 | 0,0359 | 0,0499 |
| C · QR-Lesion | **cqr_norm** | 0,8328 | 0,8948 | **0,8597** | 0,0409 | 0,0782 |

O CQR normalizado, aplicado aos mesmos modelos B/C, eleva a cobertura em lesão de 0,728/0,737 para 0,848/0,860 — fecha e ultrapassa a do ResM (0,816) — e simultaneamente aproxima a cobertura global do alvo de 0,90 (0,876 para 0,894). Como a única diferença em relação ao CQR aditivo é o esquema de calibração (multiplicativo por `w(x)` vs aditivo constante), conclui-se que a vantagem do ResM observada no S5 é atribuível à **adaptatividade local da calibração**, não à arquitetura. Hipótese de mecanismo confirmada.

### 4.2 Item 2 — fronteira de eficiência (cobertura a largura igualada)

A leitura no nível 0.90 do item 4.1 esconde uma nuance que a fronteira de eficiência revela: o CQR normalizado só atinge cobertura alta porque alarga substancialmente os intervalos em lesão (Width_lesion 0,0725/0,0782 vs 0,0555 do ResM). Comparando a cobertura à largura igualada, a ordenação é **ResM > CQR aditivo > CQR normalizado**: para qualquer largura de intervalo em lesão, o ResM atinge a maior cobertura empírica. A figura `figures/efficiency_frontier.png` mostra a curva do ResM acima das demais em todo o eixo de largura.

A interpretação é que a escala local do ResM, `u(x)`, treinada para prever diretamente a magnitude do resíduo, é uma variável de condicionamento mais eficiente, em lesão, do que a largura quantílica `w(x)` usada pelo CQR normalizado. Isso é coerente com Lei et al. (2018, Fig. 7), que demonstram que o esquema localmente ponderado pode resultar em intervalos mais largos quando a escala local não casa bem com a heteroscedasticidade real. A narrativa final do achado, portanto, não é \"é só calibração\", e sim: a cobertura condicional em lesão é governada pela adaptatividade local da calibração; entre escalas locais, a magnitude de resíduo aprendida supera a largura quantílica em eficiência.

### 4.3 Item 4 — confiabilidade (nominal vs empírico)

A figura `figures/reliability_curve.png` mostra, no painel global, que os cinco calibradores acompanham bem a diagonal de calibração perfeita, com leve sub-cobertura do CQR aditivo (B/C) e calibração quase exata para o ResM e o CQR normalizado. No painel de lesão, a separação é clara em todos os níveis: as curvas do CQR normalizado (B/C) e do ResM ficam próximas da diagonal, enquanto o CQR aditivo (B/C) fica consistentemente abaixo. Isto confirma que o ganho do CQR normalizado em lesão não é um artefato do nível 0.90, mas um comportamento sistemático em toda a faixa de níveis nominais.

### 4.4 Item 3 — cobertura condicional (Clopper-Pearson, micro)

Cobertura global e em lesão (calibrador original de cada grupo), com gap condicional:

| Grupo | Coverage_global | Coverage_lesion | Gap (global − lesão) |
|---|---|---|---|
| A (ResM) | 0,8921 | 0,8164 | 7,6 pp |
| B (QR) | 0,8763 | 0,7277 | 14,9 pp |
| C (QR-Lesion) | 0,8771 | 0,7371 | 14,0 pp |

Estratificação por sequência (cobertura em lesão):

| Grupo | AXFLAIR | AXT1 | AXT1POST |
|---|---|---|---|
| A | 0,8959 | **0,7250** | 0,8635 |
| B | 0,7736 | **0,6118** | 0,8704 |
| C | 0,7726 | **0,6346** | 0,8710 |

Estratificação por carga lesional do slice (cobertura em lesão):

| Grupo | Carga baixa | Carga média | Carga alta |
|---|---|---|---|
| A | 0,8700 | 0,8921 | **0,8011** |
| B | 0,7921 | 0,7963 | **0,7134** |
| C | 0,7937 | 0,7962 | **0,7247** |

Mesmo o ResM apresenta gap condicional de ~7,6 pp, coerente com Barber et al. (2021): cobertura condicional exata é inatingível distribution-free. O padrão mais informativo é que a sub-cobertura em lesão é dominada pela sequência AXT1 (0,61–0,73 nos três grupos), enquanto AXT1POST é bem coberta (~0,87); e que a cobertura degrada com a carga lesional, sendo o estrato de carga alta o pior em todos os grupos. Tamanhos amostrais por estrato (pixels): AXFLAIR 688.549, AXT1 835.846, AXT1POST 456.826. Os IC Clopper-Pearson são reportados em `cond_cov.csv` (ver ressalva na Seção 5.2 sobre sua interpretação).

### 4.5 Item 5 — magnitude pareada (BCa + tamanho de efeito, macro por slice)

Seleção das comparações mais informativas (delta = métrica do primeiro grupo menos a do segundo; IC 95% BCa; rank-biserial; d_z):

| Par | Métrica | n | Delta | IC 95% BCa | rank-biserial | d_z |
|---|---|---|---|---|---|---|
| A-B | Coverage_lesion | 362 | +0,0850 | [0,0761; 0,0945] | 0,798 | **0,93** |
| A-B | Width_global | 736 | -0,0045 | [-0,0051; -0,0039] | -0,708 | -0,53 |
| A-B | Width_lesion | 362 | +0,0038 | [0,0026; 0,0051] | 0,239 | 0,32 |
| A-B | IoU_topk_lesion | 362 | +0,0036 | [-0,0052; 0,0126] | 0,138 | 0,04 |
| A-B | ULAS_lesion | 362 | -0,0212 | [-0,0241; -0,0184] | -0,816 | **-0,77** |
| A-C | Coverage_lesion | 362 | +0,0828 | [0,0740; 0,0923] | 0,795 | **0,92** |
| A-C | ULAS_lesion | 362 | -0,0242 | [-0,0276; -0,0210] | -0,822 | **-0,75** |
| A-C | IoU_topk_lesion | 362 | +0,0075 | [0,0026; 0,0181] | 0,221 | 0,11 |
| B-C | Coverage_lesion | 362 | -0,0022 | [-0,0031; -0,0014] | -0,232 | -0,26 |
| B-C | Width_lesion | 362 | -0,0007 | [-0,0009; -0,0004] | -0,331 | -0,27 |
| B-C | ULAS_lesion | 362 | -0,0030 | [-0,0047; -0,0014] | -0,324 | -0,19 |
| B-C | IoU_topk_lesion | 362 | +0,0040 | [-0,0003; 0,0145] | 0,106 | 0,06 |

Dois fatos estruturais. Primeiro, o ResM redistribui largura: é mais estreito globalmente (Width_global delta A-B = -0,0045, d_z -0,53) e mais largo em lesão (Width_lesion delta +0,0038), exatamente a adaptatividade local prevista pelo item 1. Segundo, e central para o enquadramento do trabalho: o efeito da loss QR-Lesion (C vs B) é direcional e confiável — IC exclui zero em Coverage_lesion, Width_lesion e ULAS_lesion (confirmando H1, H2, H3 do S5.9) — porém pequeno (d_z 0,19–0,26); enquanto o efeito da calibração (A vs CQR) em Coverage_lesion é grande (d_z ~0,92–0,93). H4 (IoU_topk_lesion) permanece indistinguível de zero em B-C (IC inclui zero).

A divergência entre cobertura e ULAS valida a métrica: em ULAS_lesion, B e C superam A (delta A-B = -0,021, d_z -0,77; delta A-C = -0,024, d_z -0,75), apesar de A cobrir mais. Cobertura e alinhamento direcional capturam aspectos distintos da qualidade da incerteza.

### 4.6 Mondrian por sequência (resultado medido)

Motivada pelo achado da Seção 4.4 (sub-cobertura concentrada em AXT1), foi testada a calibração Mondrian condicional à sequência (Vovk et al., 2005): um `q` por sequência (AXFLAIR/AXT1/AXT1POST) calibrado no split cal e avaliado no test, comparado ao esquema marginal (q único). Cobertura em lesão por sequência, micro, nível 0.90:

| Calibrador | AXFLAIR (marg → mond) | AXT1 (marg → mond) | AXT1POST (marg → mond) |
|---|---|---|---|
| A · ResM (scaled) | 0,896 → 0,896 | 0,725 → 0,725 | 0,864 → 0,863 |
| B · CQR aditivo | 0,774 → 0,802 | **0,612 → 0,579** | 0,870 → 0,842 |
| C · CQR aditivo | 0,773 → 0,801 | **0,635 → 0,605** | 0,871 → 0,842 |
| B · CQR normalizado | 0,923 → 0,922 | 0,769 → 0,771 | 0,880 → 0,881 |
| C · CQR normalizado | 0,919 → 0,919 | 0,796 → 0,798 | 0,886 → 0,887 |

Três resultados, todos negativos ou nulos quanto à hipótese de que o Mondrian por sequência corrigiria a lesão:

Primeiro, o Mondrian por sequência **não** conserta a sub-cobertura em lesão e, no CQR aditivo, **piora AXT1** (B: 0,612 → 0,579; C: 0,635 → 0,605). A razão é mecânica: o `q` de cada sequência é calibrado sobre o pool global de pixels (dominado por fundo), e o pool de AXT1 pede um `q` menor que o marginal (B: 0,0082 vs 0,0103); o intervalo mais estreito reduz ainda mais a cobertura justamente nas lesões, que já eram o ponto cego. Conclui-se que a sub-cobertura em lesão é um fenômeno intra-sequência (lesão vs fundo), não entre sequências.

Segundo, para o ResM e o CQR normalizado o Mondrian por sequência é praticamente um no-op: os `q` por sequência quase não diferem do marginal (A: 2,052-2,054 vs 2,053; cqr_norm B: 0,836-0,853 vs 0,843), enquanto no CQR aditivo eles variam cerca de 45% (AXFLAIR 0,0119 vs AXT1 0,0082). Ou seja, a calibração localmente adaptativa já absorve a heterogeneidade entre sequências — após normalizar pelo `u(x)` ou `w(x)`, os scores ficam homogêneos entre sequências e não resta heterogeneidade para o Mondrian explorar. É evidência adicional a favor do mecanismo da Seção 4.1.

Terceiro, o Mondrian corrige a calibração global entre sequências do CQR aditivo (que estava sub-coberto em AXFLAIR, 0,848, e sobre-coberto em AXT1POST, 0,949, aproximando-se de 0,874 e 0,931), mas isso apenas redistribui largura sem beneficiar a lesão. Em AXT1 a cobertura global no test caiu (0,881 → 0,845), sinal de um deslocamento cal-test específico de AXT1: o `q` ajustado na calibração subcobre no teste.

Implicação: a alavanca correta para a sub-cobertura em lesão não é condicionar por sequência, e sim (i) a adaptatividade local da calibração (Seção 4.1) ou (ii) Mondrian condicionado ao estrato lesão/fundo — que ataca a causa direta, com a ressalva de que pixels de lesão são raros (n menor, `q` mais incerto) e de que a garantia passa a ser condicional ao estrato lesão. Execução em `scripts/analyze_mondrian_coverage.py`; ilustração em `figures/mondrian_axt1.png`.

---

## 5. Ressalvas metodológicas

### 5.1 Micro vs macro

A varredura (itens 1, 2, 4) e a análise condicional (item 3) usam **micro-média**: a cobertura é a fração de pixels cobertos sobre o pool de todos os pixels (de lesão ou globais). O S5.8 e a análise pareada (item 5) usam **macro-média**: a média, entre slices, da fração de cobertura por slice. As duas são válidas e respondem a perguntas diferentes — a micro pondera por pixel, a macro pondera por slice. Isso explica por que Coverage_lesion aparece como ~0,73–0,82 (micro, este documento) e ~0,78–0,87 (macro, S5.8) para os mesmos modelos. Recomendação: adotar a **macro por slice como métrica primária para inferência** (o slice é a unidade trocável do desenho pareado) e reportar a **micro como descritiva** das curvas de calibração, declarando explicitamente a escolha.

### 5.2 Clopper-Pearson e correlação espacial

Os IC Clopper-Pearson do item 3 são muito estreitos (largura ~0,001–0,002) porque tratam os ~1,98 milhão de pixels de lesão como ensaios de Bernoulli independentes. Pixels dentro de uma mesma lesão/slice são fortemente correlacionados espacialmente, de modo que o tamanho amostral efetivo é muito menor que a contagem de pixels. Esses IC, portanto, **subestimam** a incerteza real e devem ser lidos como descritivos da proporção micro pooled, não como inferência sobre unidades independentes. Para inferência sobre as diferenças (e o gap), o objeto correto é a análise pareada por slice (item 5), cujos IC BCa respeitam a unidade trocável.

---

## 6. Interpretação e implicações para a tese

As cinco análises convergem para uma narrativa mais precisa e mais defensável que a do S5 isolado. O achado mais forte do trabalho não é a loss QR-Lesion em si, mas a demonstração mecanística de que, em UQ pixelwise para lesão, a cobertura condicional é governada pela adaptatividade local da calibração (item 1), com a ressalva de que a forma da escala local importa e que a magnitude de resíduo aprendida do ResM é a mais eficiente (item 2). A contribuição original (Grupo C) permanece válida, com efeito pequeno e confiável (item 5), e o ULAS é validado por capturar um eixo — alinhamento direcional — ortogonal à cobertura (item 5). A sub-cobertura residual concentrada em AXT1 (item 3) é um achado clínico concreto; a Seção 4.6 mostra que ela não se resolve por condicionamento à sequência, apontando para a adaptatividade local e o Mondrian por lesão como caminhos.

Para o documento final do TCC, sugere-se: (i) promover a fronteira de eficiência (item 2) a figura principal; (ii) reposicionar a discussão do achado ResM como resultado mecanístico (itens 1–2) em vez de observação incidental; (iii) declarar macro como métrica primária com a ressalva da Seção 5.

---

## 7. Trabalho futuro

- **Calibração condicional por lesão (Mondrian por estrato lesão/fundo)**: a Seção 4.6 mostrou que o Mondrian por sequência não corrige a sub-cobertura em lesão (e a piora em AXT1), por ser dominado pelos pixels de fundo. O passo natural é calibrar `q` condicionado ao estrato lesão/fundo, atacando a causa direta. Referência: Vovk et al. (2005); Romano et al. (2020). Custo: baixo a médio (recalibração, sem re-treino); ressalva: pixels de lesão são raros, então `q` em lesão é mais incerto, e a garantia passa a ser condicional ao estrato.
- **ResM com loss lesion-aware**: cruzar a escala local mais eficiente (ResM) com a ponderação por região (loss QR-Lesion) — Caminho 3 do S5. Custo: médio a alto (re-treino).
- **Inferência respeitando correlação espacial**: bootstrap em nível de slice ou de volume para o gap condicional, em vez de Clopper-Pearson por pixel (Seção 5.2).

---

## 8. Reprodutibilidade auditável

### 8.1 SHA-256 dos checkpoints (coincidem com S5.7/S5.8)

| Grupo | Checkpoint SHA-256 (best.pt) |
|---|---|
| A (ResM) | `9ef2fa8e4e85706e2c90cdedc26cfe85dbf73987c6438b6070505072b493aeda` |
| B (QR) | `fc185dedb3457b3cec41e7640af3d35e8d968d4ae5f9431a3267952da516080a` |
| C (QR-Lesion) | `fa5198f832acd58fc8f0ed00a244dd92a86a8e3d2c7b7a8e57d0b20bc8cc58fe` |

### 8.2 Datasets Kaggle de entrada

`tcc-mri-recons-varnet-brain-4x` (splits cal/test), `tcc-mri-lesion-masks` (máscaras .pt), `tcc-mri-resm-checkpoints`, `tcc-mri-qr-checkpoints`, `tcc-mri-qr-lesion-checkpoints`, `tcc-mri-s5-8-metrics` (os três CSVs por slice usados nos itens 3 e 5).

### 8.3 Totais processados

Calibração: 730 slices, 70.759.312 pixels. Test: 736 slices, 68.366.624 pixels globais, 1.981.221 pixels de lesão. Grade de níveis nominais: 0,50; 0,55; 0,60; 0,65; 0,70; 0,75; 0,80; 0,85; 0,90; 0,925; 0,95; 0,96; 0,97; 0,98; 0,99. eps = 1e-6.

### 8.4 Comandos

```bash
# Itens 3 e 5 (CPU, sobre os metrics_*.csv do S5.8)
python scripts/analyze_conditional_coverage.py --metrics-dir <dir_metrics> --output results/cond_cov.csv
python scripts/analyze_paired_deltas.py        --metrics-dir <dir_metrics> --output results/paired.csv

# Itens 1, 2, 4 (uma invocacao por grupo/calibrador; idempotente)
python scripts/analyze_calibration_sweep.py --group A --calibrator scaled   --checkpoint <A/best.pt> --recons-dir <recons> --masks-dir <masks> --output results/sweep_A_scaled.csv
python scripts/analyze_calibration_sweep.py --group B --calibrator cqr       --checkpoint <B/best.pt> --recons-dir <recons> --masks-dir <masks> --output results/sweep_B_cqr.csv
python scripts/analyze_calibration_sweep.py --group B --calibrator cqr_norm  --checkpoint <B/best.pt> --recons-dir <recons> --masks-dir <masks> --output results/sweep_B_cqr_norm.csv
python scripts/analyze_calibration_sweep.py --group C --calibrator cqr       --checkpoint <C/best.pt> --recons-dir <recons> --masks-dir <masks> --output results/sweep_C_cqr.csv
python scripts/analyze_calibration_sweep.py --group C --calibrator cqr_norm  --checkpoint <C/best.pt> --recons-dir <recons> --masks-dir <masks> --output results/sweep_C_cqr_norm.csv

# Figuras + tabela 0.90
python scripts/plot_calibration_extras.py --sweep-dir results --out-dir figures

# Mondrian por sequencia (Secao 4.6): uma invocacao por grupo/calibrador
python scripts/analyze_mondrian_coverage.py --group A --calibrator scaled --checkpoint <A/best.pt> --recons-dir <recons> --masks-dir <masks> --output results/mondrian_A_scaled.csv
python scripts/analyze_mondrian_coverage.py --group B --calibrator cqr    --checkpoint <B/best.pt> --recons-dir <recons> --masks-dir <masks> --output results/mondrian_B_cqr.csv
python scripts/analyze_mondrian_coverage.py --group C --calibrator cqr    --checkpoint <C/best.pt> --recons-dir <recons> --masks-dir <masks> --output results/mondrian_C_cqr.csv

# Orquestracao completa no Kaggle: notebooks/kaggle_S5_extras.ipynb
```

### 8.5 Versões de software

Python 3.12; PyTorch 2.10; scipy >= 1.11 (BCa via `scipy.stats.bootstrap`); pandas >= 2.0; numpy >= 1.24; matplotlib. Versões pinadas em `requirements.txt`.

### 8.6 Sementes

Bootstrap BCa do delta pareado: seed 42 (`numpy.random.default_rng(42)`). Quantis empíricos da varredura e da calibração Mondrian: determinísticos (sem aleatoriedade).

---

## 9. Referências

- Angelopoulos, A.N.; Bates, S. (2023). Conformal Prediction: A Gentle Introduction. *Foundations and Trends in Machine Learning*, 16(4):494-591.
- Barber, R.F. et al. (2021). The limits of distribution-free conditional predictive inference. *Information and Inference*, 10(2):455-482.
- Clopper, C.J.; Pearson, E.S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika*, 26(4):404-413.
- Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7:1-30.
- Efron, B.; Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
- Kerby, D.S. (2014). The simple difference formula: an approach to teaching nonparametric correlation. *Comprehensive Psychology*, 3:1.
- Lei, J. et al. (2018). Distribution-Free Predictive Inference for Regression. *Journal of the American Statistical Association*, 113(523):1094-1111.
- Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile Regression. *NeurIPS*, 32:3543-3553.
- Romano, Y. et al. (2020). Classification with Valid and Adaptive Coverage. *NeurIPS*.
- Vovk, V.; Gammerman, A.; Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.

---

*Documento construído sobre as saídas reais executadas em 04/06/2026 e 06/06/2026 (`summary_level090.csv`, `cond_cov.csv`, `paired.csv`, `sweep_*.csv`, `mondrian_*.csv`), geradas pelos scripts da branch feat/S5-extras. Os q_hat recomputados reproduzem a calibração S5.7, validando o pipeline.*
