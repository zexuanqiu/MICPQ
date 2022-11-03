# MICPQ

A Pytorch implementation of paper "**Efficient Document Retrieval by End-to-End Refining and Quantizing BERT Embedding with Contrastive Product Quantization **" (EMNLP 2022).

### Main Dependencies

- pytorch 1.7.1
- transformers 4.24.0

### How to Run

```shell
# An example. 
# Run on the NYT Ddataset, 16-bit setting
CUDA_VISIBLE_DEVICES=0 python main.py ./checkpoint/nyt16 ./data/nyt --train --cuda --seed 0 --prob_weight 0.1 --cond_ent_weight 0.1 --L_word 24 --N_books 4 --N_words 16 --batch_size 64 --epochs 100 --lr 0.001 --encode_length 16 --max_length 400  --gumbel_temperature 10.0 --dist_metric euclidean --code_weight 1.0 

```

Also, one can refer to the `run.sh` for detailed running commands to reproduce the results reported in our paper.