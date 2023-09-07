import os
import argparse
from pipe import *
import sentencepiece as spm
from transformers import LlamaTokenizer, AutoTokenizer
from tokenization_chatglm import ChatGLMTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


def count_language_tokens(vocab, lang='ch'):
    # vocab = tokenizer.get_vocab()
    print(f'{len(vocab)=}')
    match lang:
        case 'ch':
            tokens = [*vocab | where(lambda x: x and all(0x4E00 <= ord(i) <= 0x9FA5 for i in x))]
        case 'jp':
            tokens = [*vocab | where(lambda x: x and all((0x3040 <= ord(i) <= 0x309F or 0x30A0 <= ord(i) <= 0x30FF) for i in x))]
        case _:
            tokens = []
    print(tokens, '\n', len(tokens))


# parser = argparse.ArgumentParser()
# parser.add_argument('--origin', default='chatglm', type=str)
# parser.add_argument('--tokenizer_path', default='../../models/chatglm2-6b', type=str)
# parser.add_argument('--sp_model_path', default='../../models/Llama-2-7b-chat-hf/tokenizer.model', type=str)
# args = parser.parse_args()
#
# origin = args.origin
# origin_tokenizer_path = args.tokenizer_path
# target_sp_model_path = args.sp_model_path
#
# origin_tokenizer = (LlamaTokenizer if origin == 'llama' else AutoTokenizer).from_pretrained(origin_tokenizer_path, trust_remote_code=True)
# origin_spm = sp_pb2_model.ModelProto()
# # pro = origin_tokenizer.sp_model.serialized_model_proto() `for llama`
# origin_spm.ParseFromString(origin_tokenizer.tokenizer.sp_model.serialized_model_proto())
#
# sp_model = spm.SentencePieceProcessor()
# sp_model.Load(target_sp_model_path)
# extend_spm = sp_pb2_model.ModelProto()
# extend_spm.ParseFromString(sp_model.serialized_model_proto())
# print(f'{len(origin_tokenizer)=}', f'{len(sp_model)=}')
# print(f'{origin_tokenizer.all_special_tokens=}')
# print(f'{origin_tokenizer.all_special_ids=}')
# print(f'{origin_tokenizer.special_tokens_map=}')
#
# origin_tokens = set(p.piece for p in origin_spm.pieces)
# for p in extend_spm.pieces:
#     piece = p.piece
#     if piece not in origin_tokens:
#         new_p = sp_pb2_model.ModelProto().SentencePiece()
#         new_p.piece = piece
#         new_p.score = 0
#         origin_spm.pieces.append(new_p)
# print(f'origin pieces: {len(origin_tokenizer)}  cur pieces: {len(origin_spm.pieces)}')
#
sp_save_path = '../../models/merged_tokenizer_sp'
hf_tokenizer_path = '../../models/merged_tokenizer'
# os.makedirs(sp_save_path, exist_ok=True)
# with open(sp_save_path + '/tokenizer.model', 'wb') as f:
#     f.write(origin_spm.SerializeToString())
# tokenizer = ChatGLMTokenizer(vocab_file=sp_save_path + '/tokenizer.model')
# tokenizer.save_pretrained(hf_tokenizer_path)
# print(f'Merged tokenizer saved at `{hf_tokenizer_path}`')

merged_tokenizer = ChatGLMTokenizer.from_pretrained(hf_tokenizer_path)
text = '白日依山尽，黄河入海流。欲穷千里目，更上一层楼。The primary use of LLaMA is research on large language models, including'
# print('Test text:\n', text)
# print(f'Origin tokenizer:{origin_tokenizer.tokenize(text)}')
# print(f'Merged tokenizer:{merged_tokenizer.tokenize(text)}')

print(merged_tokenizer.tokenize(' a' * 100))

