# tcc-mri-uncertainty

Pixel-wise uncertainty quantification for accelerated brain MRI reconstruction using conformalized quantile regression with lesion-aware loss weighting. Built on E2E-VarNet and fastMRI. — Data Science \& Analytics





\## Dependências externas (não versionadas)



Antes de rodar os scripts, clone o repositório oficial do fastMRI+ em `data/`:



\\`\\`\\`bash

cd data/

git clone https://github.com/microsoft/fastmri-plus.git

\\`\\`\\`



O `brain.csv` em `data/fastmri-plus/Annotations/brain.csv` é referenciado pelos

scripts em `scripts/`. Ref: Zhao et al. (2022) \*Scientific Data\* 9:152.



Os dados brutos do fastMRI brain multicoil (.h5) devem ser baixados separadamente

de https://fastmri.med.nyu.edu/ após registro. Caminho esperado nos scripts:

o que você configurar (ex.: `D:\\Mri\\anotados\\`).

