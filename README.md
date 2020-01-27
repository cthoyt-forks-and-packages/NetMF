# NetMF

This is a Python implementation of NetMF for the task of network embedding learning, as described in our paper:
 
[Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec](https://arxiv.org/abs/1710.02971)

## Data Sets


BlogCatalog [Source](http://socialcomputing.asu.edu/datasets/BlogCatalog3) [Preprocessed](http://leitang.net/code/social-dimension/data/blogcatalog.mat)

Protein-Protein Interaction [Source](http://thebiogrid.org/download.php) [Preprocessed](http://snap.stanford.edu/node2vec/Homo_sapiens.mat)

Wikipedia [Source](http://www.mattmahoney.net/dc/textdata) [Preprocessed](http://snap.stanford.edu/node2vec/POS.mat)

[Flickr](http://leitang.net/code/social-dimension/data/flickr.mat)

## Installation

The latest version of NetMF can be downloaded and installed from
[GitHub](https://github.com/xptree/NetMF) on Python 3.5+ with:

```bash
$ pip install git+https://github.com/xptree/NetMF.git
```

For developers, NetMF can be installed from the source code in
in development mode with:

```bash
$ git clone https://github.com/xptree/NetMF.git
$ cd NetMF
$ pip install -e .
```

## Usage

This is a minimal example to use the code via the command line to train on the PPI network.

```bash
$ wget http://snap.stanford.edu/node2vec/Homo_sapiens.mat
$ netmf-train --input Homo_sapiens.mat --output homo_sapiens_embeddings.txt
$ netmf-predict --label Homo_sapiens.mat --embedding homo_sapiens_embeddings.txt.npy --seed 5
```

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{qiu2018network,
  title={Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec},
  author={Qiu, Jiezhong and Dong, Yuxiao and Ma, Hao and Li, Jian and Wang, Kuansan and Tang, Jie},
  booktitle={Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining},
  pages={459--467},
  year={2018},
  organization={ACM}
}
```
