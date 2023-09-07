import os
import platform

import torch

from utils import load_model_on_gpus
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, LlamaModel
from peft import PeftModel


tokenizer = AutoTokenizer.from_pretrained('../../../models/chatglm2-6b', trust_remote_code=True)
# model = AutoModel.from_pretrained('../../../models/chatglm2-6b', trust_remote_code=True)
model_1 = torch.load('../outputs/2023-08-30/15-01-01/lora_model/epoch_9_step_1063/pytorch_model.bin')
print(model_1)
exit()

# print(sum(p.numel() for p in model.parameters()))
# for p in model.parameters():
#     print(p.name, p.size(), p.numel())
# for name, para in model.named_parameters():
#     print(f'{name:} {para.size()}')

model = PeftModel.from_pretrained(model, '../outputs/2023-08-30/15-01-01/lora_model/epoch_9_step_1063')
# model = load_model_on_gpus('../../../models/chatglm2-6b', num_gpus=2)
model = model.eval()
clear_command = 'cls' if platform.system() == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = '欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序'
    for query, response in history:
        prompt += f'\n\n用户：{query}'
        prompt += f'\n\nChatGLM2-6B：{response}'
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    past_key_values, history = None, []
    global stop_stream
    print('欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序')
    while True:
        query = input('\n用户：')
        if query.strip() == 'stop':
            break
        if query.strip() == 'clear':
            past_key_values, history = None, []
            os.system(clear_command)
            print('欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序')
            continue
        print('\nChatGLM：', end='')
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end='', flush=True)
                current_length = len(response)
        print('')


if __name__ == '__main__':
    main()
