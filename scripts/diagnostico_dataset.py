# Autor: Massanori
# Data: 13/05/2026
# Descrição: Script diagnóstico para investigar volumes específicos. Renderiza
#            uma sequência de fatias adjacentes de um único volume com todas
#            as bboxes anotadas para identificar padrões metodológicos
#            (ex.: flip vertical, lesões secundárias, anotações em fatias
#            vizinhas). Foi usado para confirmar a necessidade da correção
#            y_fastmri = H - y_dicom - h no bbox_to_mask durante a validação
#            da S3. Recebe: pasta anotados/, brain.csv, nome do volume.
#            Retorna: figura PNG com 5 fatias lado a lado + estatísticas no console.


"""
diagnostico_dataset.py
======================
Diagnóstico do split fastMRI+ — verifica quais arquivos têm bbox real
vs. apenas anotações study-level, e avalia o impacto no split do S3.

Uso:
    python scripts/diagnostico_dataset.py \
        --brain_csv caminho/para/brain.csv \
        --split_dir caminho/para/data/  \
        --hdf5_dir  D:/Mri/anotados/

Referências:
    Zhao et al. (2022). fastMRI+: Clinical Pathology Annotations for Knee
    and Brain Fully Sampled Multi-Coil MRI Data. Scientific Data 9:152.
    https://doi.org/10.1038/s41597-022-01255-z
"""

import argparse
import os
import numpy as np
import pandas as pd


# ── Argumentos ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Diagnóstico do dataset fastMRI+")
    p.add_argument("--brain_csv",  required=True,
                   help="Caminho para fastmri-plus/Annotations/brain.csv")
    p.add_argument("--split_dir",  default=None,
                   help="Pasta onde estão train.npy, val.npy, cal.npy, test.npy "
                        "(opcional — se não informado, pula análise do split)")
    p.add_argument("--hdf5_dir",   default=None,
                   help="Pasta com os arquivos .h5 baixados (opcional — "
                        "se informado, verifica quais arquivos existem em disco)")
    return p.parse_args()


# ── Utilitários ───────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── 1. Leitura e inspeção básica ──────────────────────────────────────────────

def inspecionar_csv(df):
    section("1. ESTRUTURA DO brain.csv")
    print(f"  Linhas totais       : {len(df):,}")
    print(f"  Colunas             : {df.columns.tolist()}")

    # Detectar coluna study-level (pode variar entre versões do fastMRI+)
    candidatas = [c for c in df.columns
                  if 'study' in c.lower() or 'level' in c.lower()]
    print(f"  Candidatas study-level: {candidatas}")
    return candidatas


# ── 2. Separar bbox real vs. study-level ─────────────────────────────────────

def classificar_arquivos(df):
    section("2. CLASSIFICAÇÃO POR TIPO DE ANOTAÇÃO")

    # Detectar coluna de coordenadas (x, X, col_x …)
    col_x = next((c for c in df.columns if c.lower() == 'x'), None)
    if col_x is None:
        print("  ⚠ Coluna 'x' não encontrada — verifique os nomes das colunas acima.")
        return None, None, None

    # Arquivos com pelo menos 1 bbox real (x não-nulo)
    bbox_rows = df[df[col_x].notna()]
    files_com_bbox = set(bbox_rows['file'].unique()
                         if 'file' in df.columns
                         else bbox_rows.iloc[:, 0].unique())

    # Detectar coluna 'file'
    col_file = 'file' if 'file' in df.columns else df.columns[0]
    todos_files = set(df[col_file].unique())
    files_sem_bbox = todos_files - files_com_bbox

    print(f"  Total de arquivos únicos    : {len(todos_files):>6}")
    print(f"  Com pelo menos 1 bbox real  : {len(files_com_bbox):>6}  ✅")
    print(f"  SEM nenhuma bbox (só study) : {len(files_sem_bbox):>6}  "
          + ("⚠" if files_sem_bbox else "✅ nenhum — split está correto"))

    if files_sem_bbox:
        print("\n  Exemplos de arquivos sem bbox:")
        for f in list(files_sem_bbox)[:5]:
            print(f"    {f}")

    return files_com_bbox, files_sem_bbox, col_file


# ── 3. Análise do split salvo ─────────────────────────────────────────────────

def analisar_split(split_dir, files_sem_bbox, col_file):
    section("3. IMPACTO NO SPLIT DO S3")

    splits = {}
    for nome in ["train", "val", "cal", "test"]:
        for ext in [".npy", ".txt"]:
            path = os.path.join(split_dir, f"{nome}{ext}")
            if os.path.exists(path):
                if ext == ".npy":
                    arr = np.load(path, allow_pickle=True)
                else:
                    with open(path) as f:
                        arr = [l.strip() for l in f if l.strip()]
                splits[nome] = list(arr)
                break

    if not splits:
        # Tentar npz único
        for fname in os.listdir(split_dir):
            if fname.endswith(".npz"):
                data = np.load(os.path.join(split_dir, fname), allow_pickle=True)
                splits = {k: list(data[k]) for k in data.files}
                break

    if not splits:
        print("  ⚠ Nenhum arquivo de split encontrado em", split_dir)
        print("    Esperados: train.npy, val.npy, cal.npy, test.npy  (ou split.npz)")
        return

    total_split = sum(len(v) for v in splits.values())
    print(f"  Tamanho do split encontrado:")
    for nome, arr in splits.items():
        contaminados = [f for f in arr if f in files_sem_bbox]
        flag = f"  ⚠ {len(contaminados)} sem bbox" if contaminados else "  ✅"
        print(f"    {nome:6s}: {len(arr):4d} volumes{flag}")

    print(f"\n  Total no split: {total_split} (esperado: 563 ou 750)")

    # Resumo de risco
    test_contaminados = [f for f in splits.get("test", []) if f in files_sem_bbox]
    if test_contaminados:
        print(f"\n  🔴 RISCO ALTO: {len(test_contaminados)} volumes sem bbox no TEST SET")
        print("     Coverage_lesion e ULAS serão NaN para esses volumes.")
        print("     Recomendação: refazer split filtrando apenas files_com_bbox.")
    else:
        print("\n  🟢 Test set limpo — métricas de lesão não serão afetadas.")


# ── 4. Verificar arquivos em disco ────────────────────────────────────────────

def verificar_disco(df, hdf5_dir, col_file):
    section("4. ARQUIVOS EM DISCO vs. brain.csv")

    arquivos_csv = set(df[col_file].unique())
    h5_em_disco  = {f for f in os.listdir(hdf5_dir) if f.endswith(".h5")}

    # Normalizar nomes (brain.csv pode ter ou não extensão)
    def norm(nome):
        return nome if nome.endswith(".h5") else nome + ".h5"

    csv_normalizados = {norm(f) for f in arquivos_csv}
    presentes   = csv_normalizados & h5_em_disco
    ausentes    = csv_normalizados - h5_em_disco
    extras      = h5_em_disco - csv_normalizados

    print(f"  Arquivos no brain.csv        : {len(arquivos_csv)}")
    print(f"  Arquivos .h5 em disco        : {len(h5_em_disco)}")
    print(f"  Presentes em ambos           : {len(presentes)}  ✅")
    print(f"  No CSV mas ausentes em disco : {len(ausentes)}"
          + ("  ⚠" if ausentes else "  ✅"))
    print(f"  Em disco mas fora do CSV     : {len(extras)}"
          + ("  (ignorar — não serão usados)" if extras else "  ✅"))

    if ausentes:
        print("\n  Primeiros ausentes em disco:")
        for f in sorted(ausentes)[:5]:
            print(f"    {f}")


# ── 5. Estatísticas de lesão ──────────────────────────────────────────────────

def estatisticas_lesao(df):
    section("5. ESTATÍSTICAS DE LESÃO (para documentação do TCC)")

    col_x = next((c for c in df.columns if c.lower() == 'x'), None)
    col_w = next((c for c in df.columns if c.lower() == 'w'), None)
    col_h = next((c for c in df.columns if c.lower() == 'h'), None)

    if not all([col_x, col_w, col_h]):
        print("  ⚠ Colunas x/w/h não detectadas.")
        return

    bbox_df = df[df[col_x].notna()].copy()
    bbox_df['area'] = bbox_df[col_w].astype(float) * bbox_df[col_h].astype(float)

    SMALL, MEDIUM = 50, 200
    bbox_df['faixa'] = pd.cut(
        bbox_df['area'],
        bins=[0, SMALL, MEDIUM, float('inf')],
        labels=['pequena (<50px²)', 'média (50-200px²)', 'grande (>200px²)']
    )

    print(f"  Total de bboxes reais: {len(bbox_df):,}")
    print(f"\n  Distribuição por faixa de área:")
    print(bbox_df['faixa'].value_counts().to_string())
    print(f"\n  Área mín / mediana / máx: "
          f"{bbox_df['area'].min():.0f} / "
          f"{bbox_df['area'].median():.0f} / "
          f"{bbox_df['area'].max():.0f} px²")

    n_pequenas_bbox = bbox_df[bbox_df['faixa'] == 'pequena (<50px²)']
    if len(n_pequenas_bbox) < 20:
        print(f"\n  ⚠ Apenas {len(n_pequenas_bbox)} lesões pequenas — "
              "use bootstrap CI (Seção 5.3 do guia) para essa faixa.")
    else:
        print(f"\n  ✅ Faixa pequena com n={len(n_pequenas_bbox)} — "
              "suficiente para testes estatísticos.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  DIAGNÓSTICO DO DATASET fastMRI+")
    print("  Referência: Zhao et al., Scientific Data 9:152 (2022)")
    print("="*60)

    # Leitura
    print(f"\n  Lendo: {args.brain_csv}")
    df = pd.read_csv(args.brain_csv)

    inspecionar_csv(df)
    files_com_bbox, files_sem_bbox, col_file = classificar_arquivos(df)

    if files_com_bbox is None:
        print("\n  Diagnóstico interrompido — verifique os nomes das colunas.")
        return

    if args.split_dir and os.path.isdir(args.split_dir):
        analisar_split(args.split_dir, files_sem_bbox, col_file)
    else:
        section("3. IMPACTO NO SPLIT DO S3")
        print("  --split_dir não informado ou não encontrado — pulando.")

    if args.hdf5_dir and os.path.isdir(args.hdf5_dir):
        verificar_disco(df, args.hdf5_dir, col_file)
    else:
        section("4. ARQUIVOS EM DISCO vs. brain.csv")
        print("  --hdf5_dir não informado — pulando.")

    estatisticas_lesao(df)

    section("RESUMO FINAL")
    print(f"  Arquivos únicos no brain.csv : {len(df['file'].unique() if 'file' in df.columns else df.iloc[:,0].unique()):>6}")
    print(f"  Com bbox real                : {len(files_com_bbox):>6}")
    print(f"  Sem bbox (study-level only)  : {len(files_sem_bbox):>6}")
    if not files_sem_bbox:
        print("\n  ✅ CONCLUSÃO: Todos os 750 arquivos têm bbox real.")
        print("     O S3 está correto. Atualize apenas a documentação (563→750).")
    else:
        print(f"\n  ⚠ CONCLUSÃO: {len(files_sem_bbox)} arquivos sem bbox detectados.")
        print("     Veja seção 3 acima para avaliar se o split precisa ser refeito.")


if __name__ == "__main__":
    main()