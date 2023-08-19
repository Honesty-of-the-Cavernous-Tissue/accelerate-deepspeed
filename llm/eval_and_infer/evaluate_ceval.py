import glob
import json
import torch
import os.path
import collections
from tqdm import tqdm
from utils import load_model_on_gpus
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel


def c_eval(data_path, model_name_or_path, batch_size, extraction_prompt, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    # model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).cuda()
    model = load_model_on_gpus(model_name_or_path, num_gpus=2)
    choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in 'ABCD']
    acc_total, count_total, submit_dict = 0., 0, collections.defaultdict(dict)
    with torch.no_grad():
        for entry in glob.glob(data_path, recursive=True):
            with open(entry, encoding='utf-8') as file:
                dataset = [json.loads(line) for line in file.readlines()]
            correct, task_name = 0, os.path.basename(entry).replace('.jsonl', '')
            dataloader = DataLoader(dataset, batch_size=batch_size)
            for batch in tqdm(dataloader):
                ids, texts = batch['id'], batch['inputs_pretokenized']
                inputs = [f'[Round 1]\n\n问：{text}\n\n答：' for text in texts]
                inputs = tokenizer(inputs, padding=True, return_tensors='pt', truncation=True, max_length=2048).to('cuda')
                outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
                outputs = [tokenizer.decode(out[len(inputs['input_ids'][i]):]) for i, out in enumerate(outputs.tolist())]

                inputs = [text + out + '\n' + extraction_prompt for text, out in zip(texts, outputs)]
                inputs = [f'[Round 1]\n\n问：{text}\n\n答：' for text in inputs]
                inputs = tokenizer(inputs, padding=True, return_tensors='pt', truncation=True, max_length=2048).to('cuda')
                outputs = model(**inputs, return_last_logit=True)

                logits = outputs.logits[:, -1][:, choice_tokens]
                preds = logits.argmax(dim=-1)
                correct += sum(preds.cpu() == batch['label']).cpu().item()
                for i, pre in zip(ids.cpu().numpy(), preds.cpu().numpy()):
                    submit_dict[task_name][str(i)] = 'ABCD'[pre]
            print(f'{task_name}: {correct / len(dataset)}')
            acc_total += correct
            count_total += len(dataset)
    with open(os.path.basename(model_name_or_path) + '_' + save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(submit_dict, ensure_ascii=False, indent=4))
    print(f'Final CEval score: {acc_total / count_total}')


if __name__ == '__main__':
    c_eval('../../../data/CEval/test/**/*.jsonl', '../../../models/chatglm2-6b', 16, '综上所述，ABCD中正确的选项是：', 'tax.json')
