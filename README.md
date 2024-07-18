# FairNormalization

The code for the paper entitled **Equitable Artificial Intelligence for Glaucoma Screening with Fair Identity Normalization** (https://www.medrxiv.org/content/10.1101/2023.12.13.23299931v1.full.pdf).

<img src="fig/framework.png" width="600">

# Requirements

To install the prerequisites, run:

```
pip install - r requirements.txt
```

# Experiments

To run the experiments with the baseline models on 2D RNFLT maps, execute:
```
./scripts/train_glaucoma_fair_npj.sh
```

To run the experiments with the baseline models with the proposed FIN module on 3D OCT B-scans, execute:
```
./scripts/train_glaucoma_fair_proposed_npj.sh
```

# Citation

Shi, Min, Yan Luo, Yu Tian, Lucy Q. Shen, Tobias Elze, Nazlee Zebardast, Mohammad Eslami et al. "Equitable Artificial Intelligence for Glaucoma Screening with Fair Identity Normalization." medRxiv (2023): 2023-12.

# Licence

Apache License 2.0

