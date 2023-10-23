import os
import timeit
import datetime
from collections import defaultdict


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def bar(iterator, title='bar'):
    """ wrap the iterator as a progress-bar """
    from alive_progress import alive_it
    return alive_it(iterator, title=title, bar='bubbles', spinner='horizontal')


class Colorful:
    def __call__(self, content, mode=7, front_color=40, back_color=';40', align='^', width=30, unit='gb', rounding=2):
        """ Align_map: { '<': 'left', '>': 'right', '^': 'mid' } \n
            Mode map: {0: 'Normal', 1: 'Bold/Highlight', 7: 'Reversed'} \n
            Front color map: {30: 'Black', 32: 'Green', 34: 'Blue', 36: 'Cyan', 40: 'White'} \n
            Back color map: {';40': 'Black', ';42': 'Green', ..., ';47': 'White'}  set `back_color` with '' to be default \n
            Font map: {7: 'normal', 1: 'bold' ...} """
        aligned = '\033[{}{' + f':{align + str(width)}' + '}\033[0m'
        if type(content) is float:
            rounded = '{' + f':.{rounding}' + 'f}'
            return aligned.format(f'{mode};{front_color}{back_color}m', rounded.format(content) + unit)
        return aligned.format(f'{mode};{front_color}{back_color}m', content)

    def red(self, content: str):
        return self.__call__(content, mode=1, front_color=31, back_color='', align='<', width=0)

    def green(self, content: str):
        return self.__call__(content, mode=1, front_color=32, back_color='', align='<', width=0)

    def yellow(self, content: str):
        return self.__call__(content, mode=1, front_color=33, back_color='', align='<', width=0)

    def blue(self, content):
        return self.__call__(content, mode=1, front_color=34, back_color='', align='<', width=0)

    def white(self, content):
        return self.__call__(content, mode=1, front_color=37, back_color='', align='<', width=0)

    def timer(self, start):
        hours, seconds = divmod(timeit.default_timer() - start, 3600)
        setting = {'mode': '1;7', 'rounding': 0, 'front_color': 36}
        hour_minute = self.__call__(hours, width=15, align='>', unit='h ', **setting)
        hour_minute += self.__call__(seconds / 60, width=15, align='<', unit='min', **setting)
        return self.__call__('Time consume: ', **setting) + hour_minute


def bytes2gigabytes(x):
    """ Converting Bytes to Megabytes """
    return x / 2 ** 30


def deep_dict(n):
    if n == 1:
        return lambda: defaultdict(set)
    return lambda: deep_dict(n - 1)


def auto_configure_device_map(num_gpus, device_map, num_trans_layers):
    per_gpu_layers = (num_trans_layers + 2) / num_gpus
    # from https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    if not device_map:
        device_map = {'transformer.embedding.word_embeddings': 0, 'transformer.encoder.final_layernorm': 0, 'transformer.output_layer': 0,
                      'transformer.rotary_pos_emb': 0, 'lm_head': 0}
        patten = 'transformer.encoder.layers'
    else:
        patten = 'model.layers'

    used, gpu_target = 2, 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'{patten}.{i}'] = gpu_target
        used += 1
    return device_map


def load_model_on_gpus(model, num_gpus=2, device_map=None, num_trans_layers=28):
    if num_gpus < 2 and device_map is None:
        return model.half()
    else:
        from accelerate import dispatch_model
        model = model.half()
        device_map = auto_configure_device_map(num_gpus, device_map, num_trans_layers)
        model = dispatch_model(model, device_map=device_map)
    return model


def check_model_para(model):
    print(model.state_dict)
    for name, para in model.named_parameters():
        print(f'{name:} {para.size()}', f'{para.requires_grad=}')


# from itertools import islice
# def chunk(it, size):
#     return iter(lambda: tuple(islice(iter(it), size)), ())

# print([*chunk([[1] * 4 for _ in range(2)], 2)])
# a = [1, 2, 3, 4]
# a = [i for i in a for _ in range(1)]
# print(a)
# # print(lambda: tuple(islice(iter(it), size)), ())
# b = list(chunk(a, 2))

# print(b)
# print(next(b))
# print(next(b))
# for i in chunk(a, 2):
#     print(i)
# import json
# records = ['a', 'b', 'c', 'd']
# with open('tmp.jsonl', 'w', encoding='utf-8') as f:
#     for record in records:
#         f.write(json.dumps({record}, ensure_ascii=False) + '\n')
# import simdjson
#
# print(simdjson.dumps({'a'}))
