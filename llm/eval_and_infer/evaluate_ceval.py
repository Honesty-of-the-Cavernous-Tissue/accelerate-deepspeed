import os
import glob
import json
import torch
import hydra
import collections
from tqdm import tqdm
from peft import PeftModel
from torch.utils.data import DataLoader
from utils import load_model_on_gpus, colorful
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, LlamaForCausalLM


@hydra.main('../config', 'CEval')
def c_eval(cfg):
    if cfg.pretrain_model_name == 'llama':
        device_map = {'model.embed_tokens': 0, 'model.norm': 0, 'lm_head': 0}
        tokenizer = LlamaTokenizer.from_pretrained(cfg.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        init_args = {'device_map': 'auto', 'torch_dtype': torch.float16}
        if cfg.quantize == 4:
            init_args['load_in_4bit'] = True
        elif cfg.quantize == 8:
            init_args['load_in_8bit'] = True
        model = LlamaForCausalLM.from_pretrained(cfg.model_name_or_path, **init_args)
    else:
        device_map = None
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
        if cfg.quantize == 4:
            model = AutoModel.from_pretrained(cfg.model_name_or_path, trust_remote_code=True).quantize(4).cuda()
        elif cfg.quantize == 8:
            model = AutoModel.from_pretrained(cfg.model_name_or_path, trust_remote_code=True).quantize(8).cuda()
        else:
            model = AutoModel.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)

    if cfg.peft_model_path:
        model = PeftModel.from_pretrained(model, cfg.peft_model_path)
    if cfg.num_gpus > 1:
        model = load_model_on_gpus(model, cfg.num_gpus, device_map, cfg.num_trans_layers)

    choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in 'ABCD']
    acc_total, count_total, tax_accountant, submit_dict = 0., 0, 0., collections.defaultdict(dict)
    with torch.no_grad():
        for entry in tqdm(glob.glob(cfg.data_path, recursive=True)):
            with open(entry, encoding='utf-8') as file:
                dataset = [json.loads(line) for line in file.readlines()[:2]]
            correct, task_name = 0, os.path.basename(entry).replace('.jsonl', '')
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size)
            for batch in dataloader:
                ids, texts = batch['id'], batch['inputs_pretokenized']
                inputs = [f'[Round 1]\n\n问：{text}\n\n答：' for text in texts]
                inputs = tokenizer(inputs, padding=True, return_tensors='pt', truncation=True, max_length=2048).to('cuda')
                outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
                outputs = [tokenizer.decode(out[len(inputs['input_ids'][i]):]) for i, out in enumerate(outputs.tolist())]

                inputs = [text + out + '\n' + cfg.extraction_prompt for text, out in zip(texts, outputs)]
                inputs = [f'[Round 1]\n\n问：{text}\n\n答：' for text in inputs]
                inputs = tokenizer(inputs, padding=True, return_tensors='pt', truncation=True, max_length=2048).to('cuda')
                if cfg.pretrain_model_name == 'llama':
                    outputs = model(**inputs)
                else:
                    outputs = model(**inputs, return_last_logit=True)

                logits = outputs.logits[:, -1][:, choice_tokens]
                preds = logits.argmax(dim=-1)
                correct += sum(preds.cpu() == batch['label']).cpu().item()
                for i, pre in zip(ids.cpu().numpy(), preds.cpu().numpy()):
                    submit_dict[task_name][str(i)] = 'ABCD'[pre]
            if task_name == 'tax_accountant':
                tax_accountant = f'{task_name}: {correct / len(dataset)}'
            acc_total += correct
            count_total += len(dataset)

    save_dir = '_'.join([os.path.basename(cfg.model_name_or_path), os.path.basename(cfg.peft_model_path), str(cfg.quantize)])
    with open(save_dir + '_' + cfg.save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(submit_dict, ensure_ascii=False, indent=4))
    print(colorful(f'Tax_accountant score: {tax_accountant}', mode=1, front_color=34, back_color=''))
    print(colorful(f'Final CEval score: {acc_total / count_total}', mode=1, front_color=34, back_color=''))


if __name__ == '__main__':
    c_eval()
