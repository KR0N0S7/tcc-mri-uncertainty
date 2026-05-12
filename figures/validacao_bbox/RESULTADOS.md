# Validação visual bbox ↔ reconstrução — fastMRI+ brain

**Data:** 2026-05-12
**Seed:** 42
**Inspeção por:** Rogério Massanori
**Refs:** Zhao et al. (2022) *Scientific Data* 9:152; Zbontar et al. (2020) *arXiv:1811.08839*; Sriram et al. (2020) *MICCAI*

## Procedimento

Validação de alinhamento entre bounding boxes do fastMRI+ e reconstruções RSS do fastMRI multicoil brain, executada em três etapas:

1. **Inspeção inicial:** 10 overlays amostrados estratificadamente por sequência (4 AXFLAIR, 4 AXT1, 2 AXT1POST) sobre `reconstruction_rss` em orientação nativa, sem nenhuma correção.

2. **Diagnóstico do volume suspeito:** investigação adicional do volume `file_brain_AXFLAIR_200_6002493` em fatias adjacentes (s5, s6, s7) revelou padrão consistente de bboxes em y ∈ [180, 287] no CSV, enquanto a lesão hiperintensa correspondente aparece em y < 140 da `reconstruction_rss`.

3. **Smoke test pós-correção:** verificação visual da máscara gerada após aplicação da transformação `y_fastmri = H - y_dicom - h`, com 100% de alinhamento sobre a lesão alvo (Figura `smoke_test_mask.png`).

## Causa-raiz identificada

O `reconstruction_rss` armazenado no HDF5 do fastMRI está em orientação espelhada no eixo Y em relação à convenção DICOM usada pelos radiologistas durante a anotação no MD.ai. O script `ExampleScripts/fastmri-to-dicom.py` do fastMRI+ aplica essa reflexão antes da geração dos DICOMs anotados.

## Correção aplicada

Em `src/data/lesion_masks.py::bbox_to_mask`, transformação `y_fastmri = H - y_dicom - h` antes da geração da máscara binária. Imagem mantida na orientação nativa fastMRI para preservar compatibilidade com a VarNet pré-treinada (Sriram et al., 2020). Correção protegida por 7 testes unitários (`tests/test_lesion_masks.py`) — todos passaram em 12/05/2026 às 01:12.

## Resultados pós-correção (10/10)

| # | Volume | Fatia | Label(s) | Veredito |
|---|--------|-------|----------|----------|
| 0 | AXFLAIR_200_6002493 | 6 | Edema + Extra-axial mass (×2) | OK (massa coberta) |
| 1 | AXFLAIR_200_6002467 | 6 | Nonspecific WM lesion | OK |
| 2 | AXFLAIR_200_6002487 | 4 | Nonspecific WM lesion (×2) | OK |
| 3 | AXFLAIR_201_6003003 | 8 | Nonspecific lesion (×2) | OK (shape 320×260) |
| 4 | AXT1_202_2020570 | 0 | Edema | OK |
| 5 | AXT1_202_2020389 | 3 | Posttreatment change | OK |
| 6 | AXT1_202_2020143 | 11 | Posttreatment change | OK |
| 7 | AXT1_202_2020509 | 12 | Craniotomy (×4) + Posttreatment (×2) | OK |
| 8 | AXT1POST_201_6002812 | 1 | Dural thickening + artefato + Posttreatment | OK |
| 9 | AXT1POST_201_6002803 | 1 | Enlarged ventricles | OK |

## Limitação documentada

Bounding boxes axis-aligned superestimam a área de tecido patológico em lesões com geometria arredondada (massas, edema vasogênico). Estimativa visual: 15-20% dos pixels marcados em massas grandes correspondem a tecido sadio adjacente. Tratado no Cap. 5 (Limitações) e como Trabalho Futuro (segmentação fina).

## Decisão

- Correção `apply_y_flip=True` é o default permanente em `bbox_to_mask`
- Aprovado para geração das máscaras de todos os 724 volumes
- Commit ref: `v0.3.2-bbox-flip-fix`