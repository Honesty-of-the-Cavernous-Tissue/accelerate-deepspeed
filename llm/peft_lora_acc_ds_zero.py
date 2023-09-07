if True:
    import logging
    logging.basicConfig(level=logging.ERROR)
import os
import gc
import json
import glob
import hydra
import torch
import psutil
import timeit
import datetime
import warnings
import threading
import itertools
from tqdm import tqdm
from datasets import Dataset
from keras.utils import Progbar
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils import Colorful, bytes2gigabytes
from accelerate import logging as ac_logging
from deepspeed.utils import logger as ds_logging
from tokenization_chatglm import ChatGLMTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from accelerate.utils import DummyOptim, DummyScheduler
from transformers import AutoTokenizer, set_seed, AutoModel, AutoConfig, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
# from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live, estimate_zero3_model_states_mem_needs_all_cold

# ac_logging.info('main only', main_process_only=True)
colorful = Colorful()
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


def init_model(cfg, accelerator, model_name_or_path, pretrain_model_name):
    init_args = {'low_cpu_mem_usage': True} if accelerator.state.deepspeed_plugin.zero_stage == 2 else {}
    match pretrain_model_name:
        case 'chatglm':
            model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, empty_init=False, **init_args)
        case 'baichuan':
            if accelerator.state.deepspeed_plugin.zero_stage == 2:
                init_args['device_map'] = 'auto'
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16, **init_args)
        case 'llama':
            # if accelerator.state.deepspeed_plugin.zero_stage == 2:
            #     init_args['device_map'] = 'auto'
            model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, **init_args)
        case _:
            raise ValueError('Invalid model name')

    if cfg.use_flash_attention:
        # from optimum.bettertransformer import BetterTransformer
        # model = BetterTransformer.transform(model)
        from llama_patch import replace_attn_with_flash_attn
        replace_attn_with_flash_attn()

    tokenizer, model_config = init_tokenizer(cfg, model, accelerator, model_name_or_path, pretrain_model_name)

    if cfg.use_peft:
        lora_config = cfg.lora_config
        init_kwargs = {'r': lora_config['r'], 'lora_alpha': lora_config['lora_alpha'], 'lora_dropout': lora_config['lora_dropout'], 'target_modules': [*lora_config['target_modules'][pretrain_model_name]]}
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, **init_kwargs)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    return model, tokenizer, model_config


def init_tokenizer(cfg, model, accelerator, model_name_or_path, pretrain_model_name):
    if pretrain_model_name == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        model_config = LlamaConfig.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if cfg.merge_tokens:
        merged_tokenizer = ChatGLMTokenizer.from_pretrained(model_name_or_path + '/merged_tokenizer')
        vocabs = tokenizer.get_vocab()
        merged_vocabs = merged_tokenizer.get_vocab()
        new_tokens = set(merged_vocabs.keys()) - set(vocabs.keys())
        tokenizer.add_tokens([*new_tokens])
        if pretrain_model_name == 'chatglm':
            model.base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        else:
            model.resize_token_embeddings(len(tokenizer))
        accelerator.print(colorful.green(content=f'Successful resized tokens from `{len(vocabs)}` to `{len(tokenizer)}`'))
    return tokenizer, model_config


def gen_data(cfg, tokenizer, model_config, is_train=True):
    assert cfg.data_path.endswith('.json'), 'Unsupported file type'
    with (open(cfg.data_path, 'r') as f):
        line = json.load(f)
        for example in tqdm(line[:int(cfg.data_percentage * len(line))]):
            prompt, label = example[cfg.prompt_column_name], example[cfg.label_column_name]
            history = example['history'] if 'history' in example else []
            prompt = ''.join(itertools.chain.from_iterable(history)) + prompt
            prompt_ids = tokenizer.encode(prompt, max_length=cfg.max_length, truncation=True)
            label_ids = tokenizer.encode(label, max_length=cfg.max_length, truncation=True, add_special_tokens=False)
            eos_token_id = model_config.eos_token_id
            input_ids = prompt_ids + label_ids + [eos_token_id] if is_train else [eos_token_id] + prompt_ids
            if len(input_ids) > cfg.max_length and cfg.skip_overlong:
                continue
            yield {'input_ids': input_ids[:cfg.max_length], 'prompt_len': len(prompt_ids)}


def gen_from_shards(shards, cfg, tokenizer, model_config):
    for shard in shards:
        with open(shard, 'r') as f:
            for line in tqdm(f.readlines()):
                input_ids = tokenizer.encode(line, max_length=cfg.max_length, truncation=True, add_special_tokens=False)
                input_ids += [model_config.eos_token_id]
                yield {'input_ids': input_ids[:cfg.max_length], 'prompt_len': len(input_ids)}


class DataCollator:
    def __init__(self, cfg, tokenizer, model_config):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model_config = model_config

    def __call__(self, features):
        input_len = [len(feature['input_ids']) for feature in features]
        longest = max(input_len)
        inputs, labels = [], []
        for f_len, feature in sorted(zip(input_len, features), key=lambda x: -x[0]):
            input_ids = feature['input_ids']
            prompt_len = feature['prompt_len']
            label_ids = [-100] * prompt_len + input_ids[prompt_len:] + [-100] * (longest - f_len)
            input_ids += [self.model_config.pad_token_id if self.cfg.pretrain_model_name == 'llama' else self.tokenizer.pad_token_id] * (longest - f_len)
            input_ids = torch.LongTensor(input_ids)
            label_ids = torch.LongTensor(label_ids)
            labels.append(label_ids)
            inputs.append(input_ids)
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        return {'input_ids': inputs, 'labels': labels}


def init_dataloader(cfg, accelerator, tokenizer, model_config):
    with accelerator.main_process_first():
        saved_path = cfg.data_path + f'_{cfg.pretrain_model_name}_{cfg.data_percentage}.saved'
        if os.path.isdir(saved_path):
            accelerator.print(colorful.yellow(f'Using cache from {saved_path}, `gen_data | gen_from_shard` function or `tokenizer` change may raise error'))
            train_dataset = Dataset.load_from_disk(saved_path)
        else:
            if cfg.mp_data_gen:
                shards = [*glob.glob(cfg.data_path + '/*.txt', recursive=True)]
                gen_kwargs = {'shards': shards, 'cfg': cfg, 'tokenizer': tokenizer, 'model_config': model_config}
                train_dataset = Dataset.from_generator(gen_from_shards, gen_kwargs=gen_kwargs, num_proc=min(os.cpu_count(), len(shards)))
                train_dataset.save_to_disk(saved_path, num_shards=len(shards), num_proc=len(shards))
            else:
                train_dataset = Dataset.from_generator(lambda: gen_data(cfg, tokenizer, model_config))
                train_dataset.save_to_disk(saved_path)
    accelerator.wait_for_everyone()
    data_collator = DataCollator(cfg, tokenizer, model_config)
    train_dataloader = DataLoader(train_dataset, cfg.batch_size, collate_fn=data_collator, pin_memory=True)
    return train_dataloader


def save_model(cfg, model, accelerator, epoch, step, last_step):
    if step and (step + 1) % cfg.save_per_steps == 0 or step == last_step:
        accelerator.wait_for_everyone()
        if not cfg.use_peft and accelerator.state.deepspeed_plugin.zero_stage == 3:
            model.save_checkpoint(cfg.model_save_path + f'epoch_{epoch + 1}_step_{step + 1}')
            accelerator.print(colorful.green('Successful saved by `DeepSpeed ZeRO-3` in `FP32`, need to be converted manually'))

        else:
            model.save_pretrained(cfg.model_save_path + f'epoch_{epoch + 1}_step_{step + 1}')
        accelerator.wait_for_everyone()


# @hydra.main(config_path='config', config_name='training', version_base='1.3')
@hydra.main(config_path='config', config_name='training')
def main(cfg):
    set_seed(cfg.seed)
    accelerator = Accelerator(log_with='wandb')
    with accelerator.main_process_first():
        accelerator.init_trackers(f'{os.path.basename(cfg.model_name_or_path)}-{os.path.basename(cfg.data_path)}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%m")}')
    model_name_or_path, pretrain_model_name = cfg.model_name_or_path, cfg.pretrain_model_name
    model, tokenizer, model_config = init_model(cfg, accelerator, model_name_or_path, pretrain_model_name)
    train_dataloader = init_dataloader(cfg, accelerator, tokenizer, model_config)

    optimizer = DummyOptim(model.parameters(), lr=cfg.lr)
    lr_scheduler = DummyScheduler(optimizer, num_warmup_steps=0, num_training_steps=(len(train_dataloader) * cfg.num_epochs))
    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(model, train_dataloader, optimizer, lr_scheduler)

    for epoch in range(cfg.num_epochs):
        with TorchTraceMemAlloc(accelerator):
            model.train()
            total_loss = 0
            pro_bar = Progbar(len(train_dataloader), unit_name='sample')
            for i, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs[0] if pretrain_model_name == 'baichuan' else outputs.loss
                pro_bar.update(i + 1, [('loss', loss.detach().float().cpu().numpy())])
                accelerator.log({'loss': loss.detach().float()})
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                save_model(cfg, model, accelerator, epoch, i, len(train_dataloader) - 1)

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(colorful.white(f'{epoch = }: {train_ppl = } {train_epoch_loss = }'))

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
