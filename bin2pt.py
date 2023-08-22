from model import ModelArgs, Transformer
import struct
import numpy as np
import torch
from argparse import ArgumentParser
import dataclasses

# Changes from: https://github.com/softmax1/llama2.c/compare/master%40%7B30day%7D...master

def load(filepath='model.bin', out_dir = 'out'):
    f = open(filepath, 'rb')
    def deserialize(t):
        d = t.detach().cpu().view(-1).numpy().astype(np.float32)
        bytes = f.read(len(d) * 4)
        b = struct.unpack(f'{len(d)}f', bytes)
        t.data.copy_(torch.from_numpy(np.array(b).reshape(t.shape)))

    p = ModelArgs(multiple_of=32)
    header_fmt = 'iiiiiii'
    header = f.read(len(header_fmt)*4)
    p.dim, hidden_dim, p.n_layers, p.n_heads, \
        p.n_kv_heads, p.vocab_size, p.max_seq_len = struct.unpack(header_fmt, header)

    model = Transformer(p)

    # next write out the embedding weights
    deserialize(model.tok_embeddings.weight)
    # now all the layers
    # attention weights
    for layer in model.layers:
        deserialize(layer.attention_norm.weight)
    for layer in model.layers:
        deserialize(layer.attention.wq.weight)
    for layer in model.layers:
        deserialize(layer.attention.wk.weight)
    for layer in model.layers:
        deserialize(layer.attention.wv.weight)
    for layer in model.layers:
        deserialize(layer.attention.wo.weight)
    # ffn weights
    for layer in model.layers:
        deserialize(layer.ffn_norm.weight)
    for layer in model.layers:
        deserialize(layer.feed_forward.w1.weight)
    for layer in model.layers:
        deserialize(layer.feed_forward.w2.weight)
    for i, layer in enumerate(model.layers):
        deserialize(layer.feed_forward.w3.weight)
    # final rmsnorm
    deserialize(model.norm.weight)
    # note: no need to write final classifier weights due to weight sharing
    # freqs_cis
    deserialize(model.freqs_cis.real[:p.max_seq_len])
    deserialize(model.freqs_cis.imag[:p.max_seq_len])

    # write to binary file
    torch.save({'model': model.state_dict(), 'model_args': dataclasses.asdict(p)}, f'{out_dir}/ckpt.pt')

parser = ArgumentParser()
parser.add_argument('--bin', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    load(args.bin, args.out_dir)