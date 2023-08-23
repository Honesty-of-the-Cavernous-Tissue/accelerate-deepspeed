

def colorful(content, mode=7, front_color=40, back_color=';40', align='^', width=30, unit='gb', rounding=2):
    """ Align_map: {'<': 'left', '>': 'right', '^': 'mid'} \n
        Mode map: {0: 'Normal', 1: 'Bold/Highlight', 7: 'Reversed'} \n
        Front color map: {30: 'Black', 32: 'Green', 34: 'Blue', 36: 'Cyan', 40: 'White'} \n
        Back color map: {';40': 'Black', ';42': 'Green', ..., ';47': 'White'}  set `back_color` with '' to be default \n
        Font map: {7: 'normal', 1: 'bold' ...} """
    aligned = '\033[{}{' + f':{align + str(width)}' + '}\033[0m'
    if type(content) is float:
        rounded = '{' + f':.{rounding}' + 'f}'
        return aligned.format(f'{mode};{front_color}{back_color}m', rounded.format(content) + unit)
    return aligned.format(f'{mode};{front_color}{back_color}m', content)


def bytes2gigabytes(x):
    """ Converting Bytes to Megabytes """
    return x / 2 ** 30


def auto_configure_device_map(num_gpus, device_map, num_trans_layers):
    per_gpu_layers = (num_trans_layers + 2) / num_gpus
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
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
