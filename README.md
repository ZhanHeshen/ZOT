# ZOT: Unlocking Black-Box Prompt Tuning Efficiency via Zeroth-Order Optimization
This is the code repository of paper Unlocking Black-Box Prompt Tuning Efficiency via Zeroth-Order Optimization

# How to use
The environment can be set up by the provided `requirements.txt` file

```
cd ZOT
pip install requirements.txt
```
We provide the modelling file of LLaMA2, GPT2 and RoBERTa-Large. You can runing experiments on these model by specifying `model_path`.


Runing the following command to run the experiments:

`python wbt.py --model_name model_path`

