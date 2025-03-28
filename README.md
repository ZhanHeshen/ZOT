# ZOT: Unlocking Black-Box Prompt Tuning Efficiency via Zeroth-Order Optimization
This is the code repository of paper Unlocking Black-Box Prompt Tuning Efficiency via Zeroth-Order Optimization [[paper link](https://aclanthology.org/2024.findings-emnlp.871.pdf)].

# Overview
Prompt optimization emerges as an important technique for adapting Large Language Models (LLMs) to specific tasks. Unfortunately, LLM proprietors often limit access to models’ internal weights, confining users to inference API services. This restriction poses a significant challenge for prompt optimization, as conventional optimization-based algorithms rely heavily on gradient information, which is unavailable via inference APIs. Addressing this challenge, this paper presents the ZerothOrder Tuning (ZOT) approach, which enables efficient prompt tuning solely via inference APIs. ZOT adopts the zeroth-order optimization framework, utilizing finite differences to approximate gradient information. We further incorporate ZOT with gradient clipping and
momentum techniques to enhance the tuning effectiveness. 

# How to use
The environment can be set up by the provided `requirements.txt` file

```
conda create --name zot python=3.10
conda activate zot
pip install fastNLP==0.6.0
pip install datasets
pip install scikit-learn
pip install transformers==4.28.1
pip install matplotlib
git clone https://github.com/ZhanHeshen/ZOT.git
cd ZOT
```
We provide the modelling file of LLaMA2, GPT2 and RoBERTa-Large. You can runing experiments on these model by specifying `model_path`.


Runing the following command to run the experiments:

`python ZOT.py --model_name model_path`

# Acknowledgements
Our code is heavily based on the [BBT](https://github.com/txsun1997/Black-Box-Tuning). Please follow the detailed instructions from BBT.

# Bibtex
If you find this code is helpful, please cite our paper in the following format.
```
@inproceedings{zhan2024unlocking,
  title={Unlocking Black-Box Prompt Tuning Efficiency via Zeroth-Order Optimization},
  author={Zhan, Heshen and Chen, Congliang and Ding, Tian and Li, Ziniu and Sun, Ruoyu},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
  pages={14825--14838},
  year={2024}
}
```
