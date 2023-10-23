import os
import glob
import json
import torch
import hydra
import timeit
import deepspeed
import collections
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from peft import PeftModel
from utils import Colorful
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM


def init_tokenizer_and_model(cfg):
    if cfg.pretrain_model_name == 'chatglm':
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(cfg.model_name_or_path, trust_remote_code=True, low_cpu_mem_usage=True)
        if cfg.quantize == 4:
            model = model.quantize(4)
        elif cfg.quantize == 8:
            model = model.quantize(8)
        if not cfg.use_ddp and not cfg.use_ds:
            model = model.cuda()
    else:
        if cfg.pretrain_model_name == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(cfg.model_name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        init_args = {'torch_dtype': torch.float16, 'low_cpu_mem_usage': True}
        if not cfg.use_ddp and not cfg.use_ds:
            init_args['device_map'] = 'auto'
        if cfg.quantize == 4:
            init_args['load_in_4bit'] = True
        elif cfg.quantize == 8:
            init_args['load_in_8bit'] = True
        if cfg.pretrain_model_name == 'llama':
            model = LlamaForCausalLM.from_pretrained(cfg.model_name_or_path, **init_args)
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, trust_remote_code=True, **init_args)
    if cfg.peft_model_path:
        model = PeftModel.from_pretrained(model, cfg.peft_model_path)
    if cfg.use_ds:
        model = deepspeed.init_inference(model, tensor_parallel={'tp_size': cfg.num_gpus}, dtype=torch.float16, kernel_inject=True)
    return tokenizer, model


def init_dataset(cfg):
    dataset = []
    for entry in glob.glob(cfg.data_path, recursive=True):
        task_name = os.path.basename(entry).replace('.jsonl', '')
        with open(entry, encoding='utf-8') as file:
            for line in file.readlines():
                line = json.loads(line)
                line['task_name'] = task_name
                dataset.append(line)
    count_total, count_tax = len(dataset), sum(i['task_name'] == 'tax_accountant' for i in dataset)
    return dataset, count_total, count_tax


def batch_fn(dataloader, tokenizer, model, rank='cuda'):
    tax_correct = all_correct = 0
    submit_dict = collections.defaultdict(dict)
    choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in 'ABCD']
    with torch.no_grad():
        for batch in tqdm(dataloader):
            ids, texts, task_names = batch['id'], batch['inputs_pretokenized'], batch['task_name']
            inputs = [f'[Round 1]\n\n问：{text}\n\n答：' for text in texts]
            inputs = tokenizer(inputs, padding=True, return_tensors='pt', truncation=True, max_length=2048).to(rank)
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
            outputs = [tokenizer.decode(out[len(inputs['input_ids'][i]):]) for i, out in enumerate(outputs.tolist())]

            inputs = [text + out + '\n' + config.extraction_prompt for text, out in zip(texts, outputs)]
            inputs = [f'[Round 1]\n\n问：{text}\n\n答：' for text in inputs]
            inputs = tokenizer(inputs, padding=True, return_tensors='pt', truncation=True, max_length=2048).to(rank)
            if config.pretrain_model_name == 'llama':
                outputs = model(**inputs)
            else:
                outputs = model(**inputs, return_last_logit=True)
            logits = outputs.logits[:, -1][:, choice_tokens]
            preds = logits.argmax(dim=-1)
            # correct += sum(preds.cpu() == batch['label']).cpu().item()
            for i, pre, label, name in zip(ids.cpu().numpy(), preds.cpu().numpy(), batch['label'].cpu().numpy(), task_names):
                submit_dict[name][str(i)] = 'ABCD'[pre]
                all_correct += pre == label
                tax_correct += pre == label and name == 'tax_accountant'
    return submit_dict, tax_correct, all_correct


def runtime_summery(count_tax, count_all, tax_correct, all_correct, submit_dict):
    save_dir = '_'.join([os.path.basename(config.model_name_or_path), os.path.basename(config.peft_model_path), str(config.quantize)])
    with open(save_dir + '_' + config.save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(submit_dict, ensure_ascii=False, indent=4))
    hour, second = divmod(timeit.default_timer() - start_time, 3600)
    print(colorful(f'Time consume: {int(hour)}h {second / 60}min', mode=1, front_color=34, back_color=''))
    print(colorful('Tax_accountant score: {:.3f}'.format(tax_correct / count_tax), mode=1, front_color=34, back_color=''))
    print(colorful('Final 52 CEval score: {:.3f}'.format(all_correct / count_all), mode=1, front_color=34, back_color=''))


def ddp_dataloader(rank, world_size, dataset):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader


def run_inference(rank):
    dist.init_process_group('nccl', rank=rank, world_size=config.world_size)
    dataset, count_total, count_tax = init_dataset(config)
    dataloader = ddp_dataloader(rank, config.world_size, dataset)
    tokenizer, model = init_tokenizer_and_model(config)
    model.to(rank)
    batch_fn(dataloader, tokenizer, model, rank)


def ddp_c_eval(cfg):
    mp.spawn(run_inference, nprocs=cfg.world_size, join=True)


# @hydra.main('../config', 'CEval', '1.3')
def c_eval(cfg):
    tokenizer, model = init_tokenizer_and_model(cfg)
    dataset, count_all, count_tax = init_dataset(cfg)
    submit_dict, tax_correct, all_correct = batch_fn(DataLoader(dataset, cfg.batch_size), tokenizer, model)
    if torch.distributed.get_rank() == 0:
        runtime_summery(count_tax, count_all, tax_correct, all_correct, submit_dict)


def acc_c_eval(cfg):
    accelerator = Accelerator()
    tokenizer, model = init_tokenizer_and_model(cfg)
    dataset, count_all, count_tax = init_dataset(cfg)
    dataloader = DataLoader(dataset, cfg.batch_size)

    model, dataloader = accelerator.prepare(model, dataloader)

    batch_fn(dataloader, tokenizer, model)


colorful = Colorful()
start_time = timeit.default_timer()
config = OmegaConf.load('../config/CEval.yaml')
config.peft_model_path = '../outputs/chatglm2-6b-tax/lora_model/epoch_4_step_1063'

if __name__ == '__main__':
    if config.use_ddp:
        ddp_c_eval(config)
    else:
        c_eval(config)
