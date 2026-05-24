# S5.8 — Metricas por Regiao + ULAS (split test, alpha=0.10)

## Metadata por grupo

| Grupo | Slices total | Slices com lesao | q_hat | Checkpoint SHA-256 |
|---|---|---|---|---|
| A (ResM) | 736 | 362 | 2.052745 | `9ef2fa8e4e85706e...` |
| B (QR) | 736 | 362 | 0.010275 | `fc185dedb3457b3c...` |
| C (QR-Lesion) | 736 | 362 | 0.010269 | `fa5198f832acd58f...` |

## Comparativo de metricas (mean ± std (n_valid))

| Metrica | A (ResM) | B (QR) | C (QR-Lesion) |
|---|---|---|---|
| Coverage_global | 0.8923 ± 0.0371 (n=736) | 0.8753 ± 0.0920 (n=736) | 0.8761 ± 0.0891 (n=736) |
| Coverage_lesion | 0.8687 ± 0.0672 (n=362) | 0.7836 ± 0.1119 (n=362) | 0.7858 ± 0.1082 (n=362) |
| Width_global | 0.0312 ± 0.0152 (n=736) | 0.0357 ± 0.0069 (n=736) | 0.0360 ± 0.0074 (n=736) |
| Width_lesion | 0.0488 ± 0.0220 (n=362) | 0.0450 ± 0.0107 (n=362) | 0.0456 ± 0.0111 (n=362) |
| IoU_topk_global | 0.1654 ± 0.0473 (n=736) | 0.1619 ± 0.0469 (n=736) | 0.1597 ± 0.0469 (n=736) |
| IoU_topk_lesion | 0.0809 ± 0.0736 (n=362) | 0.0773 ± 0.0679 (n=362) | 0.0734 ± 0.0473 (n=362) |
| ULAS_lesion | 0.5820 ± 0.0723 (n=362) | 0.6032 ± 0.0563 (n=362) | 0.6062 ± 0.0536 (n=362) |
| ULAS_null | 0.5668 ± 0.0617 (n=362) | 0.5885 ± 0.0452 (n=362) | 0.5916 ± 0.0422 (n=362) |
| ULAS_z_score | 2.6500 ± 2.9962 (n=362) | 2.7660 ± 3.1694 (n=362) | 2.6911 ± 3.1595 (n=362) |

## Notas

- `Coverage` reportado como macro-average por slice. Cobertura nominal alvo: 1 - alpha = 0.90.
- `Width` em escala normalizada por `max_val` do volume (D1).
- `IoU_topk` com X = 5% (top-X% mais incertos vs top-X% com maior erro).
- `ULAS_null` = media de 10 permutacoes do error_map; `ULAS_z_score = (ULAS_real - ULAS_null) / std_null`.
- Para n_valid das metricas globais: total de slices test. Para metricas de lesao: total de slices com pelo menos 1 pixel de lesao.

## Próximo passo (S5.9)

Analise estatistica formal sobre os CSVs por slice:
- **Friedman test + Nemenyi post-hoc** (Demšar, 2006) para comparações entre os 3 grupos.
- **Holm-Bonferroni** para correcao em multiplas comparacoes par-a-par.
- **BCa bootstrap** para CIs 95% das metricas agregadas.
- **Clopper-Pearson** para IC exato de Coverage como proporcao.
