# Tutorial

This repository contains the implementation for our paper "**Self-supervised Topic Taxonomy Discovery in the Box Embedding Space**" accepted to TACL.

In summary, we develop a Box embedding-based Topic Model (BoxTM) that maps words and topics into the box embedding space, where the asymmetric metric is defined to properly infer hierarchical relations among topics. Additionally, our BoxTM explicitly infers upper-level topics based on correlation between specific topics through recursive clustering on topic boxes.



## 0. Environment

```
conda create -n BoxTM_env python=3.7.13
conda activate BoxTM_env
pip install -r requirements.txt
```



## 1. Evaluation

The corpora used in this work (i.e., 20NS, NYT, and arXiv) can be downloaded [here](https://drive.google.com/file/d/1hy75h5zoQKYE9OT8FlxUnFt_whMpbL-k/view?usp=sharing), and the corresponding pretrained checkpoints are provided in the `model` folder, i.e., `model/[dataset]/all_model_pretrain.*`.

You can run the script of `scripts/eval.sh` to load the pretrained checkpoints and  reproduce the results of our BoxTM on the intrinsic evaluation.

```
bash scripts/eval.sh
```

The results are saved in `log/[dataset]_pretrain_test.log`. The output format is as follows:

```
BoxTM model loaded from ./model/nyt_50_8171/all_model_pretrain.pkl.
# Top-30 keywords of the leaf topics
[Level-0 Topic]
Topic 0:['film', 'films', 'movie', 'characters', 'movies', 'actors', 'hollywood', 'cinema', 'comedy', 'dvd', 'festival', 'actor', 'filmmakers', 'actress', 'studio', 'documentary', 'theater', 'character', 'mr', 'comic', 'oscar', 'musical', 'directed', 'audience', 'starring', 'viewers', 'ms', 'producer', 'romantic', 'music']
...
# keywords of the leaf topics.
# e.g., "Topic 0 [0, 25]" means that Topic (0-)0 and Topic (0-)25 at level 0 are children of Topic (1-)0 at level 1.
[Level-1 Topic]
Topic 0 [0, 25]:['music', 'festival', 'film', 'opera', 'album', 'songs', 'mr', 'orchestra', 'musical', 'dance', 'films', 'musicians', 'ballet', 'characters', 'movie', 'artists', 'movies', 'theater', 'song', 'piano', 'dancers', 'guitar', 'audience', 'band', 'character', 'singer', 'ms', 'premiere', 'studio', 'art']
...
[Level-2 Topic]
Topic 0 [0, 13, 15]:['music', 'film', 'festival', 'mr', 'theater', 'movie', 'opera', 'song', 'dance', 'films', 'movies', 'songs', 'audience', 'album', 'characters', 'art', 'ms', 'musicians', 'musical', 'band', 'studio', 'artists', 'ballet', 'orchestra', 'company', 'hollywood', 'singer', 'dancers', 'artist', 'piano']
...
# Topic coherence scores of top-10 keywords
Topic coherence: [Level-0(47)] 0.4062 [Level-1(20)] 0.3987 [Level-2(8)] 0.4043 [all] 0.4040
# (reported) Average CLNPMI scores of top-5, top-10, and top-15 keywords
clnpmi: [Level-0(47)] 0.1807 [Level-1(20)] 0.1686 [all] 0.1771
# (reported) Average coherence scores of top-5, top-10, and top-15 keywords
Ave Topic coherence: [Level-0(47)] 0.4109 [Level-1(20)] 0.4033 [Level-2(8)] 0.4119 [all] 0.4090
# (reported) Average TU scores of top-5, top-10, and top-15 keywords
Ave TU: [Level-0(47)] 0.8955 [Level-1(20)] 0.9717 [Level-2(8)] 0.9903 [all] 0.6476
```



## 2. Training

Here we provide a example of training the BoxTM model on NYT from scratch.

**a. Configure the hyperparameter settings in the `BoxTM_config.yaml` file.** 

```yaml
# [BoxTM_config.yaml]
nyt:
  dataset: 'nyt'
  save_name: 'model.pkl'
  input_type: "tfidf"
  batch_size:  256
  epochs: 2000
  hidden_num: 256
  emb_size: 50
  learning_rate: 5.e-3
  level_num: 3
  topic_nums: 50
  emb_type: 'glove'
  freeze_word: False
  vocab_num: 8171
  init_word_scale:
    - 0.0001
    - 0.01
    - 0.9
    - 0.999
  init_topic_scale:
    - 0.0001
    - 0.01
    - 0.9
    - 0.999
  intersect_temp: 0.1
  volume_temp: 0.1
  r1: 1.0
  r2: 0.5
  r3: 0.05 # weight of word box constraint
  w1: 3.0 # weight of L_{CO} (alpha)
  w2: 0.005 # weight of L_{HT} (beta)
  sample_size: 512
```



**b. Run the script of `scripts/run.sh` to start training.**

```
bash scripts/run.sh
```

The training log is saved in `log/nyt_train.log`.

Once the training is complete, the script will perform the intrinsic evaluation on the trained BoxTM model. The results are stored in `log/nyt_test.log`.



## Citation

```
@article{10.1162/tacl_a_00712,
    author = {Lu, Yuyin and Chen, Hegang and Mao, Pengbo and Rao, Yanghui and Xie, Haoran and Wang, Fu Lee and Li, Qing},
    title = "{Self-supervised Topic Taxonomy Discovery in the Box Embedding Space}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {12},
    pages = {1401-1416},
    year = {2024},
    month = {11},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00712},
    url = {https://doi.org/10.1162/tacl\_a\_00712},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00712/2478615/tacl\_a\_00712.pdf},
}
```


