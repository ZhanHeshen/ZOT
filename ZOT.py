import os
import copy
import time
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
import argparse
import numpy as np
import pandas as pd
from fastNLP import cache_results, Tester, DataSet, DataSetIter, RandomSampler, SequentialSampler
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    GPT2Config,
    GPT2Tokenizer,
    LlamaConfig,
    LlamaTokenizer,
)
from models.modeling_roberta import RobertaForMaskedLM
from models.modeling_gpt2 import GPT2LMHeadModel
from models.modeling_llama import LlamaForCausalLM
from utils import hinge_loss
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tasks import get_task


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='llama-7b-hf', type=str)
parser.add_argument("--task_name", default='sst2', type=str)
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--k_shot", default=16, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--budget", default=2000, type=int)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=100, type=int)
parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--loss_type", default='ce', type=str)
parser.add_argument("--cat_or_add", default='add', type=str)
parser.add_argument("--learning_rate", default=0.1, type=float)
parser.add_argument("--lr_min", default=0.01, type=float)
parser.add_argument("--clip_grad", default=100, type=float)
parser.add_argument("--momentum", default=0.96, type=float)
parser.add_argument("--zo_eps", default=1e-1, type=float)
parser.add_argument("--log_dir", default="logs", type=str)
parser.add_argument("--n_point", default=1, type=int)
parser.add_argument(
    "--inference_framework",
    default='pt',
    type=str,
    help='''Which inference framework to use. 
         Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively'''
)
args = parser.parse_args()
print(args)
model_name = args.model_name

if 'gpt2' in model_name:
    from dataloaders.dataloader_gpt import SST2Loader, BoolQLoader, CoLALoader, COPALoader, AGNewsLoader, YelpPLoader, DBPediaLoader, RTELoader, MRPCLoader, SNLILoader
    from metrics.metrics_gpt import SST2Metric, BoolQMetric, CoLAMetric, COPAMetric, AGNewsMetric, YelpPMetric, DBPediaMetric, RTEMetric, MRPCMetric, SNLIMetric
elif 'roberta' in model_name:
    from dataloaders.dataloader import SST2Loader, BoolQLoader, CoLALoader, COPALoader, AGNewsLoader, YelpPLoader, DBPediaLoader, RTELoader, MRPCLoader, SNLILoader, MMLULoader
    from metrics.metrics import SST2Metric, BoolQMetric, CoLAMetric, COPAMetric, AGNewsMetric, YelpPMetric, DBPediaMetric, RTEMetric, MRPCMetric, SNLIMetric
elif 'llama' in model_name:
    from dataloaders.dataloader_gpt import SST2Loader, CoLALoader, COPALoader, AGNewsLoader, YelpPLoader, DBPediaLoader, RTELoader, MRPCLoader, SNLILoader
    from metrics.metrics_gpt import SST2Metric, AGNewsMetric, YelpPMetric, DBPediaMetric, RTEMetric, MRPCMetric, SNLIMetric



task_name = args.task_name
n_prompt_tokens = args.n_prompt_tokens
k_shot = args.k_shot
batch_size = args.batch_size
budget = args.budget

device = args.device
seed = args.seed
loss_type = args.loss_type
print_every = args.print_every
eval_every = args.eval_every

cat_or_add = args.cat_or_add
inference_framework = args.inference_framework

if task_name in ['sst2', 'yelpp', 'rte', 'mrpc', 'boolq', 'cola', 'copa']:
    num_labels = 2
elif task_name in ['agnews']:
    num_labels = 4
elif task_name in ['dbpedia']:
    num_labels = 14
args.bbt_version = 'wbt'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# =============== load dataset ==================
if 'roberta' in model_name:
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
elif 'gpt2' in model_name:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
elif 'llama' in model_name:
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

cache_fn = f"caches/data_{model_name.replace('/', '-')}_{task_name}_{n_prompt_tokens}_{seed}.pt"
DataLoader = {
    'sst2': SST2Loader,
    # 'boolq': BoolQLoader,
    # 'cola': CoLALoader,
    # 'copa': COPALoader,
    'agnews': AGNewsLoader,
    'yelpp': YelpPLoader,
    'dbpedia': DBPediaLoader,
    'rte': RTELoader,
    'mrpc': MRPCLoader,
    'snli': SNLILoader,
}



@cache_results(cache_fn, _refresh=False)
def get_data(task_name, tokenizer):
    if task_name in ['agnews', 'yelpp', 'dbpedia', 'snli']:
        splits = ['train', 'test']
    else:  # for datasets without test set, we use dev set
        splits = ['train', 'validation']
    if args.cat_or_add == 'cat':
        data_bundle = DataLoader[task_name](tokenizer=tokenizer, n_prompt_tokens=0).my_load(splits)
    else:
        data_bundle = DataLoader[task_name](tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens).my_load(splits)
    return data_bundle


def construct_true_few_shot_data(train_data, k_shot):
    train_label_count = {}
    dev_label_count = {}
    new_train_data = DataSet()
    new_dev_data = DataSet()
    all_indices = [_ for _ in range(len(train_data))]
    np.random.shuffle(all_indices)

    for index in all_indices:
        if task_name == 'Copa':
            label = train_data[index].data['label']
        else:
            label = train_data[index]['labels']
        if label < 0:
            continue

        if label not in train_label_count:
            train_label_count[label] = 0
        if label not in dev_label_count:
            dev_label_count[label] = 0

        if train_label_count[label] < k_shot:
            if task_name == 'Copa':
                new_train_data.append(train_data[index].data)
            else:
                new_train_data.append(train_data[index])
            train_label_count[label] += 1
        elif dev_label_count[label] < k_shot:
            if task_name == 'Copa':
                new_dev_data.append(train_data[index].data)
            else:
                new_dev_data.append(train_data[index])
            dev_label_count[label] += 1


    if 'gpt2' in model_name:
        new_train_data.set_input("input_ids", "attention_mask")
        new_dev_data.set_input("input_ids", "attention_mask")
    elif 'llama' in model_name:
        new_train_data.set_input("input_ids", "attention_mask")
        new_dev_data.set_input("input_ids", "attention_mask")
    else:
        new_train_data.set_input("input_ids", "attention_mask", "mask_pos")
        new_dev_data.set_input("input_ids", "attention_mask", "mask_pos")

    new_train_data.set_target("labels")
    new_dev_data.set_target("labels")
    return new_train_data, new_dev_data


data_bundle = get_data(task_name=task_name, tokenizer=tokenizer)
if task_name in ['agnews', 'yelpp', 'dbpedia', 'snli']:
    train_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('test')
elif task_name == 'Copa':
    train_data, test_data = data_bundle.samples['train'], data_bundle.samples['valid']
else:
    train_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('validation')
train_data, dev_data = construct_true_few_shot_data(train_data, k_shot)
for ds in [train_data, dev_data, test_data]:
    ds.set_pad_val('input_ids', tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    ds.set_pad_val('attention_mask', 0)
print('# of train data: {}'.format(len(train_data)))
print('Example:')
print(train_data[0])
print('\n# of dev data: {}'.format(len(dev_data)))
print('Example:')
print(dev_data[0])
print('\n# of test data: {}'.format(len(test_data)))
print('Example:')
print(test_data[0])


# =============== transform data into tensor ==================
def transform_data(model_name, data):
    if 'roberta' in model_name:
        data = {
                'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
                'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
                'mask_pos': torch.tensor(train_data['mask_pos'].get(list(range(len(train_data))))),
                'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
            }
    elif 'gpt2' in model_name:
        data = {
                'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
                'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
                'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
            }
    elif 'llama' in model_name:
        data = {
                'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
                'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
                'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
            }
    return data


# =============== prepare the model ========================
if 'roberta' in model_name:
    config = RobertaConfig.from_pretrained(args.model_name)
    model = RobertaForMaskedLM.from_pretrained(
                    args.model_name,
                    config=config,
                    n_prompt_tokens=n_prompt_tokens,
                    inference_framework=inference_framework,)
elif 'gpt2' in model_name:
    config = GPT2Config.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(
                    args.model_name,
                    config=config,
                    n_prompt_tokens=n_prompt_tokens)
elif 'llama' in model_name:
    config = LlamaConfig.from_pretrained(args.model_name)
    model = LlamaForCausalLM.from_pretrained(
                    args.model_name,
                    config=config,
                    n_prompt_tokens=n_prompt_tokens,)
model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(config.vocab_size))
model.to(device)

# =============== prepare the optimizer ========================
if 'roberta' in model_name:
    hidden_dim = model.roberta.get_input_embeddings().weight.shape[1]
else:
    hidden_dim = model.transformer.get_input_embeddings().weight.shape[1]
prompt_embedding = torch.zeros(n_prompt_tokens * hidden_dim, device=device)#.requires_grad_(True)
optimizer = torch.optim.AdamW([prompt_embedding], lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget, eta_min=args.lr_min)

# ===============initialize the arguments ========================
if task_name == 'sst2':
    metric = SST2Metric(target='labels', pred='logits', tokenizer=tokenizer)
    metric_key = 'acc'
    metric_name = 'SST2Metric'
elif task_name == 'boolq':
    metric = BoolQMetric(target='labels', pred='logits', tokenizer=tokenizer)
    metric_key = 'acc'
    metric_name = 'BoolQMetric'
elif task_name == 'cola':
    metric = CoLAMetric(target='labels', pred='logits', tokenizer=tokenizer)
    metric_key = 'acc'
    metric_name = 'CoLAMetric'
elif task_name == 'copa':
    metric = COPAMetric(target='labels', pred='logits', tokenizer=tokenizer)
    metric_key = 'acc'
    metric_name = 'COPAMetric'
elif task_name == 'agnews':
    metric = AGNewsMetric(target='labels', pred='logits', tokenizer=tokenizer)
    metric_key = 'acc'
    metric_name = 'AGNewsMetric'
elif task_name == 'yelpp':
    metric = YelpPMetric(target='labels', pred='logits', tokenizer=tokenizer)
    metric_key = 'acc'
    metric_name = 'YelpPMetric'
elif task_name == 'dbpedia':
    metric = DBPediaMetric(target='labels', pred='logits', tokenizer=tokenizer)
    metric_key = 'acc'
    metric_name = 'DBPediaMetric'
elif task_name == 'rte':
    metric = RTEMetric(target='labels', pred='logits', tokenizer=tokenizer)
    metric_key = 'acc'
    metric_name = 'RTEMetric'
elif task_name == 'mrpc':
    metric = MRPCMetric(target='labels', pred='logits', tokenizer=tokenizer)
    metric_key = 'f1'
    metric_name = 'MRPCMetric'
elif task_name == 'snli':
    metric = SNLIMetric(target='labels', pred='logits', tokenizer=tokenizer)
    metric_key = 'acc'
    metric_name = 'SNLIMetric'
best_train_perf = 0.0
best_dev_perf = 0.0
dev_loss = 0.0
dev_perf = 0.0
best_prompt = None
num_call = 0
ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')


# =============== define the function ========================
def calc_metric(logits, target):
        label_map = metric.label_map

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        interest_index = list(label_map.keys())
        logits = logits[:, interest_index]
        pred = logits.argmax(dim=-1)

        if metric_key == 'acc':
            perf = (pred == converted_target).sum() / len(target)
        elif metric_key == 'f1':
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(),
                            pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f'[Metric] Only support [acc, f1], got {metric_key} instead.')

        if loss_type == 'hinge':
            loss = hinge_loss(logits, converted_target, margin=metric.margin, reduction='sum') / len(target)
        elif loss_type == 'ce':
            loss = ce_loss(logits, converted_target)
        elif loss_type == 'perf':
            loss = -1 * perf
        else:
            raise KeyError(f'[Loss] Only support [hinge, ce, perf], got {loss_type} instead.')

        return loss, perf


def zero_train(num_call, model, train_data, prompt_embedding, best_train_perf, momentum):
    learning_rate = scheduler.get_last_lr()[0]
    args.learning_rate = learning_rate
    zo_eps = args.zo_eps
    
    zero_grad = 0
    for i in range(args.n_point):
        perturb = torch.normal(mean=0.0, std=1.0, size=prompt_embedding.size(), device=device)

        prompt_embedding = prompt_embedding+zo_eps*perturb
        loss_1, perf = test(model, train_data, prompt_embedding)

        prompt_embedding = prompt_embedding-2*zo_eps*perturb
        loss_2, _ = test(model, train_data, prompt_embedding)

        prompt_embedding = prompt_embedding+zo_eps*perturb
        projected_grad = ((loss_1 - loss_2) / (2 * zo_eps))#.item()
        zero_grad += projected_grad*perturb
    zero_grad = zero_grad/args.n_point
    g_norm = torch.norm(zero_grad)
    if args.clip_grad>0:
        if g_norm>args.clip_grad:
            zero_grad = args.clip_grad*zero_grad/g_norm
    momentum = args.momentum*momentum + (1-args.momentum)*zero_grad
    m_hat = momentum/(1-args.momentum**(num_call+1))

    prompt_embedding = prompt_embedding - learning_rate * m_hat

    scheduler.step()
    if perf > best_train_perf:
        best_train_perf = perf

    return loss_1, perf, prompt_embedding, best_train_perf, momentum

def eval(model, test_data, prompt_embedding, metric_name, metric_key, metric):
    # evaluate the model on test data
    model.eval()
    prompt_embedding = prompt_embedding.reshape(n_prompt_tokens, -1).repeat(batch_size, 1, 1)
    model.set_prompt_embedding(prompt_embedding)
    test_tester = Tester(data=test_data, model=model, metrics=metric, batch_size=16,
                                 num_workers=1, device=device, use_tqdm=True)
    results = test_tester.test()
    test_acc = results[metric_name][metric_key]
    test_loss = results[metric_name]['ce']
    return test_loss, test_acc

def test(model, dev_data, prompt_embedding):
    model.eval()
    # if args.cat_or_add == 'cat':
    #     model.set_concat_prompt(True)
    # else:
    #     model.set_concat_prompt(False)
    prompt_embedding = prompt_embedding.reshape(n_prompt_tokens, -1).repeat(batch_size, 1, 1)#.requires_grad_(True)
    model.set_prompt_embedding(prompt_embedding)
    for k, v in dev_data.items():
        dev_data[k] = v.to(device)
    with torch.no_grad():
        if 'gpt2' in model_name:
            logits = model(input_ids=dev_data['input_ids'],
                        attention_mask=dev_data['attention_mask'],
                    )['logits']
        elif 'llama' in model_name:
            logits = model(input_ids=dev_data['input_ids'],
                        attention_mask=dev_data['attention_mask'],
                    )['logits']
        else:
            logits = model(input_ids=dev_data['input_ids'],
                        attention_mask=dev_data['attention_mask'],
                        mask_pos=dev_data['mask_pos'],
                    )['logits']
    dev_loss, dev_perf = calc_metric(logits, dev_data['labels'])
    return dev_loss, dev_perf

# =============== train the model ========================
model.eval()
momentum = torch.zeros(prompt_embedding.size(), device=device)
best_prompt = torch.zeros(prompt_embedding.size(), device=device)
prompt_list = []
start_time = time.time()
while num_call < args.budget:
    train_data_iterator = DataSetIter(train_data, batch_size=args.batch_size, sampler=RandomSampler(), drop_last=True)
    for data, label in train_data_iterator:
        data['labels'] = label['labels']
        loss, perf, prompt_embedding, best_train_perf, momentum = zero_train(num_call, model, data, prompt_embedding, best_train_perf, momentum)
        prompt_list.append(prompt_embedding.detach().cpu())
        num_call += 1
        if num_call % print_every == 0:
            print('[# Step {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
                    num_call,
                    round(float(loss), 4),
                    round(float(perf), 4),
                    round(float(best_train_perf), 4)))
        if num_call % eval_every == 0:
            print('********* Evaluated on dev set *********')
            dev_loss, dev_perf = eval(model, dev_data, prompt_embedding, metric_name, metric_key, metric)
            if dev_perf > best_dev_perf:
                best_dev_perf = dev_perf
                best_prompt = copy.deepcopy(prompt_embedding.detach())
            print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
                    round(float(dev_loss), 4),
                    round(float(dev_perf), 4),
                    round(float(best_dev_perf), 4)))
            print('********* Done *********')
        if not isinstance(loss, float):
            loss = loss.detach().item()
        if num_call >= args.budget:
            break

end_time = time.time()
print('Total time: {} (mins)'.format((end_time - start_time)/60 ))
print('evaluating on test set...')
teloss, teacc = eval(model, test_data, best_prompt, metric_name, metric_key, metric)
with open("all_test.txt", "a") as f:
    f.write(f"dataset:{args.log_dir}/seed:{args.seed}/loss: {teloss}/ acc:{teacc}\n")
