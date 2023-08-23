if True:
    import logging
    logging.basicConfig(level=logging.ERROR)
import os
import gc
import json
import hydra
import torch
import psutil
import timeit
import warnings
import threading
import itertools
from tqdm import tqdm
from datasets import Dataset
from keras.utils import Progbar
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils import colorful, bytes2gigabytes
from accelerate import logging as ac_logging
from deepspeed.utils import logger as ds_logging
from peft import LoraConfig, TaskType, get_peft_model
from accelerate.utils import DummyOptim, DummyScheduler
from transformers import AutoTokenizer, set_seed, AutoModel, AutoConfig, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
# from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live, \
#     estimate_zero3_model_states_mem_needs_all_cold

# ac_logging.info('main only', main_process_only=True)
ds_logging.setLevel('ERROR')
warnings.filterwarnings('ignore')
ac_logger = (ac_logging.get_logger(__name__)).setLevel('ERROR')


class TorchTraceMemAlloc:
    """ This context manager is used to track the peak memory usage of the process """
    def __init__(self, accelerator):
        self.cpu_peak = -1
        self.accelerator = accelerator

    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.gpu_begin = bytes2gigabytes(torch.cuda.memory_allocated())
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        self.time_begin = timeit.default_timer()
        return self

    def cpu_mem_used(self):
        """ get resident set size memory for the current process """
        return bytes2gigabytes(self.process.memory_info().rss)

    def peak_monitor_func(self):
        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001)
            if not self.peak_monitoring:
                break

    def memory_report(self, begin, used, peaked, map_name='GPU'):
        self.accelerator.print(colorful(f'{map_name} Memory Usage', mode='1;7', front_color=32, width=60))
        self.accelerator.print(colorful('Before training') + colorful(begin))
        self.accelerator.print(colorful('After training') + colorful(used))
        self.accelerator.print(colorful('Peak during training') + colorful(peaked))
        self.accelerator.print(colorful('Total Peak during training') + colorful(peaked + begin))

    def __exit__(self, *exc):
        self.peak_monitoring = False
        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.gpu_used = bytes2gigabytes(self.end - self.gpu_begin)
        self.gpu_peaked = bytes2gigabytes(self.peak - self.gpu_begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = bytes2gigabytes(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = bytes2gigabytes(self.cpu_peak - self.cpu_begin)

        # Printing the GPU/CPU memory usage details and time consume
        self.accelerator.print()
        hours, seconds = divmod(timeit.default_timer() - self.time_begin, 3600)
        setting = {'mode': '1;7', 'rounding': 0, 'front_color': 36}
        hour_minute = colorful(hours, width=15, align='>', unit='h ', **setting)
        hour_minute += colorful(seconds / 60, width=15, align='<', unit='min', **setting)
        self.accelerator.print(colorful('Time consume: ', **setting) + hour_minute)
        self.memory_report(self.gpu_begin, self.gpu_used, self.gpu_peaked)
        self.memory_report(self.cpu_begin, self.cpu_used, self.cpu_peaked, map_name='CPU')


@hydra.main(config_path='config', config_name='training')
def main(cfg):
    def preprocess(path, is_train, percentage=0.01):
        with (open(path, 'r') as f):
            line = json.load(f)
            for example in tqdm(line[:int(percentage * len(line))]):
                prompt, label = example[cfg.prompt_column_name], example[cfg.label_column_name]
                history = example['history'] if 'history' in example else []
                prompt = ' '.join(itertools.chain.from_iterable(history)) + ' ' + prompt
                prompt_ids = tokenizer.encode(prompt, max_length=cfg.max_length, truncation=True)
                label_ids = tokenizer.encode(label, max_length=cfg.max_length, truncation=True, add_special_tokens=False)
                eos_token_id = model_config.eos_token_id
                input_ids = prompt_ids + label_ids + [eos_token_id] if is_train else [eos_token_id] + prompt_ids
                if len(input_ids) > cfg.max_length and cfg.skip_overlong:
                    continue
                yield {'input_ids': input_ids[:cfg.max_length], 'prompt_len': len(prompt_ids)}

    def data_collator(features: list) -> dict:
        # from https://github.com/mymusise/ChatGLM-Tuning/blob/master/finetune.py
        # prompt的label设置为-100，在训练时不纳入loss的计算 or use `DataCollatorForLanguageModeling` prompt纳入计算
        len_ids = [len(feature['input_ids']) for feature in features]
        longest = max(len_ids)
        inputs, labels = [], []
        for f_len, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
            input_ids = feature['input_ids']
            prompt_len = feature['prompt_len']
            label_ids = [-100] * (prompt_len - 1) + input_ids[prompt_len - 1:] + [-100] * (longest - f_len)
            input_ids = input_ids + [model_config.pad_token_id if cfg.pretrain_model_name == 'llama' else tokenizer.pad_token_id] * (longest - f_len)
            input_ids = torch.LongTensor(input_ids)
            label_ids = torch.LongTensor(label_ids)
            labels.append(label_ids)
            inputs.append(input_ids)
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        return {'input_ids': inputs, 'labels': labels}

    set_seed(cfg.seed)
    # wandb.init(project='test', config={'epoch': cfg.num_epochs})
    accelerator = Accelerator(log_with='wandb')
    accelerator.init_trackers(f'{cfg.pretrain_model_name}-{os.path.basename(cfg.data_path)}-{timeit.default_timer()}')
    model_name_or_path = cfg.model_name_or_path
    pretrain_model_name = cfg.pretrain_model_name

    if pretrain_model_name == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        model_config = LlamaConfig.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    with accelerator.main_process_first():
        data_path = cfg.data_path
        saved_path = data_path + f'_{cfg.pretrain_model_name}_{cfg.data_percentage}.saved'
        if os.path.isdir(saved_path):
            accelerator.print(colorful(f'Using cache at {saved_path}, `preprocess` fn change may raise error', mode=1, front_color=34, back_color=''))
            train_dataset = Dataset.load_from_disk(saved_path)
        else:
            train_dataset = Dataset.from_generator(lambda: preprocess(data_path, True, cfg.data_percentage))
            train_dataset.save_to_disk(saved_path)
    accelerator.wait_for_everyone()
    train_dataloader = DataLoader(train_dataset, cfg.batch_size, True, collate_fn=data_collator, pin_memory=True)
    # print(next(iter(train_dataloader)))

    # creating model
    from_pretrained_para = {'low_cpu_mem_usage': True} if accelerator.state.deepspeed_plugin.zero_stage == 2 else {}
    assert pretrain_model_name in ['chatglm', 'llama', 'baichuan'], print(f'Not supported {pretrain_model_name=}')
    if pretrain_model_name == 'chatglm':
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, empty_init=False, **from_pretrained_para)
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    elif pretrain_model_name == 'baichuan':
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16, **from_pretrained_para)
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=['W_pack'])
    elif pretrain_model_name == 'llama':
        if accelerator.state.deepspeed_plugin.zero_stage == 2:
            from_pretrained_para['device_map'] = 'auto'
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, **from_pretrained_para)
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    else:
        raise ValueError('Invalid model name')

    if cfg.use_flash_attention:
        # from optimum.bettertransformer import BetterTransformer
        # model = BetterTransformer.transform(model)

        from llama_patch import replace_attn_with_flash_attn
        replace_attn_with_flash_attn()

    # estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    optimizer = DummyOptim(model.parameters(), lr=cfg.lr)
    lr_scheduler = DummyScheduler(optimizer, num_warmup_steps=0, num_training_steps=(len(train_dataloader) * cfg.num_epochs))
    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(model, train_dataloader, optimizer, lr_scheduler)
    # accelerator.print(model)

    for epoch in range(cfg.num_epochs):
        with TorchTraceMemAlloc(accelerator):
            model.train()
            total_loss = 0
            pro_bar = Progbar(len(train_dataloader), unit_name='step')  # stateful_metrics=['loss']
            for i, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                if pretrain_model_name == 'baichuan':
                    loss = outputs[0]
                else:
                    loss = outputs.loss
                pro_bar.update(i, [('loss', loss.detach().float().cpu().numpy())])
                # wandb.log({'loss': loss.detach().float().cpu().numpy()})
                accelerator.log({'loss': loss.detach().float().cpu().numpy()})
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if i % cfg.save_per_steps == 0 or i == len(train_dataloader) - 1:
                    accelerator.wait_for_everyone()
                    model.save_pretrained(cfg.model_save_path + f'epoch_{epoch}_step_{i}')
                    accelerator.wait_for_everyone()

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f'{epoch = }: {train_ppl = } {train_epoch_loss = }')

    # wandb.finish()
    accelerator.end_training()

    # is_ds_zero_3 = False
    # if getattr(accelerator.state, 'deepspeed_plugin', None):
    #     is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
    # model.eval()
    # eval_preds = []
    # with TorchTracemalloc() as tracemalloc:
    #     for _, batch in enumerate(tqdm(eval_dataloader)):
    #         batch = {k: v for k, v in batch.items() if k != 'labels'}
    #         with torch.no_grad():
    #             # synced_gpus=True for DS-stage 3
    #             outputs = accelerator.unwrap_model(model).generate(**batch, synced_gpus=is_ds_zero_3, max_new_tokens=200)
    #         eval_preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    #         print(f'eval_preds: {eval_preds[-1]}')


if __name__ == '__main__':
    main()
