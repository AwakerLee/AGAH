## Adaptive Graph Attention Hashing for Unsupervised Cross-modal Retrieval via Multimodal Transformers

This repository contains the author's implementation in PyTorch for the APWEB-WAIM-23 paper "Adaptive Graph Attention Hashing for Unsupervised Cross-modal Retrieval via Multimodal Transformers".

## Introduction

Unsupervised cross-modal hashing retrieval has been extensively studied due to its advantages in storage, retrieval efficiency, and label independence. However, there are still two obstacles to existing unsupervised methods: (1) Existing unsupervised methods suffer from inaccurate similarity as simple features do not describe fine-grained multimodal relationships. (2) Existing methods suffer from unbalanced multimodal learning due to the different coding capabilities of different modal networks. To address these obstacles, we devised an effective Adaptive Graph Attention Hashing (AGAH) for unsupervised cross-modal
retrieval. Firstly, we use the multimodal transformer model CLIP to extract cross-modal fine-grained features and exploit multiple data similarities to mine similar information from different perspectives in multimodal data and perform similarity enhancement. In addition, we present an adaptive graph attention hashing module to assist in generating hash codes, which uses an attention mechanism to learn relation-based similarity from image-text modality. It aggregates the essential neighborhood message of neighboring data nodes through the graph neural networks to generate more discriminative hash codes. Sufficient experiments on three benchmark datasets demonstrate that the proposed AGAH outperforms existing advanced unsupervised cross-modal hashing methods.

<div align=center><img src="https://github.com/AwakerLee/CFRH/blob/main/Fig31.jpg" width="90%" height="90%"></div align=center>

***********************************************************************************************************
## Dependencies

Please, install the following packages:

- Python (>=3.8)
- pytorch
- torchvision
- h5py
- CLIP

## Datasets
You can download the features of the datasets from:
For datasets, we follow [Deep Cross-Modal Hashing's Github (Jiang, CVPR 2017)](https://github.com/jiangqy/DCMH-CVPR2017/tree/master/DCMH_matlab/DCMH_matlab). You can download these datasets from:
- Wikipedia articles, [Link](http://www.svcl.ucsd.edu/projects/crossmodal/)
- MIRFLICKR25K, [[OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EpLD8yNN2lhIpBgQ7Kl8LKABzM68icvJJahchO7pYNPV1g?e=IYoeqn)], [[Baidu Pan](https://pan.baidu.com/s/1o5jSliFjAezBavyBOiJxew), password: 8dub]
- NUS-WIDE (top-10 concept), [[OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EoPpgpDlPR1OqK-ywrrYiN0By6fdnBvY4YoyaBV5i5IvFQ?e=kja8Kj)], [[Baidu Pan](https://pan.baidu.com/s/1GFljcAtWDQFDVhgx6Jv_nQ), password: ml4y]
 - MS-COCO, [BaiduPan(password: 5uvp)](https://pan.baidu.com/s/1uoV4K1mBwX7N1TVmNEiPgA)
 
## Implementation

Here we provide the implementation of our proposed models, along with datasets. The repository is organised as follows:

 - `data/` contains the necessary dataset files for NUS-WIDE, MIRFlickr, and MS-COCO;
 - `models.py` contains the implementation of the model;
 
 Finally, `main.py` puts all of the above together and can be used to execute a full training run on MIRFlcikr or NUS-WIDE or MS-COCO.

## Process
 - Place the datasets in `data/`
 - Set the experiment parameters in `main.py`.
 - Train a model:
 ```bash
 python main.py
```
 - Modify the parameter `EVAL = True` in `main.py` for evaluation:
  ```bash
 python main.py
```

## Citation
If you find our work or the code useful, please consider cite our paper using:
```bash
}
```
