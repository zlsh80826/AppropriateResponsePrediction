from collections import defaultdict
from gensim.models import FastText
from itertools import zip_longest
from tqdm import tqdm
import argparse

def gen_vocab_dict(embedding):
    model = FastText.load(embedding)
    index = 1
    vocab_dict = defaultdict(int)
    for vocab in model.wv.vocab:
        vocab_dict[vocab] = index
        index += 1

    return vocab_dict

def read_tsv(datadir, type_):
    data = []
    with open(datadir + '/{}.tsv'.format(type_), encoding='utf-8') as file:
        for line in file:
            if type_.find('test') < 0:
                c1, c2, y = line.split('\t')
                data.append((c1.split(' '), c2.split(' '), y[:1]))
            else:
                c1, c2 = line.split('\t')
                data.append((c1.split(' '), c2.split(' ')))
    return data

def write_ctf(datadir, type_, data, vocab_dict):
    word_size = 12
    pad_spec = '{0:<%d.%d}' % (word_size, word_size)
    sanitize = str.maketrans({"|": None, "\n": None})

    with open(datadir + '/{}.ctf'.format(type_), 'w', encoding='utf-8') as file:
        for idx, d in tqdm(enumerate(data), desc='Writing {}.ctf'.format(type_)):
            if type_.find('test') < 0:
                c1, c2, ys = d[0], d[1], d[2]
                for c1_token, c2_token, y in zip_longest(c1, c2, ys):
                    out = [str(idx)]
                    if c1_token is not None:
                        out.append('|# %s' % pad_spec.format(c1_token.translate(sanitize)))
                        out.append('|c1 {}:{}'.format(vocab_dict[c1_token], 1))
                    if c2_token is not None:
                        out.append('|# %s' % pad_spec.format(c2_token.translate(sanitize)))
                        out.append('|c2 {}:{}'.format(vocab_dict[c2_token], 1))
                    if y is not None:
                        out.append('|y %3d  ' % int(y))
                    file.write('\t'.join(out))
                    file.write('\n')
            else:
                c1, c2 = d[0], d[1]
                for c1_token, c2_token in zip_longest(c1, c2):
                    out = [str(idx)]
                    if c1_token is not None:
                        out.append('|# %s' % pad_spec.format(c1_token.translate(sanitize)))
                        out.append('|c1 {}:{}'.format(vocab_dict[c1_token], 1))
                    if c2_token is not None:
                        out.append('|# %s' % pad_spec.format(c2_token.translate(sanitize)))
                        out.append('|c2 {}:{}'.format(vocab_dict[c2_token], 1))
                    file.write('\t'.join(out))
                    file.write('\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert tsv format to ctf format')
    parser.add_argument('--datadir', help='Data directory', required=False, default='../data')
    parser.add_argument('--embedding', help='Embedding weights file path', required=False, default='../embedding/fasttext.bin')
    args = parser.parse_args()

    vocab_dict = gen_vocab_dict(args.embedding)
    train_data = read_tsv(args.datadir, 'train')
    valid_data = read_tsv(args.datadir, 'valid')
    test_data = read_tsv(args.datadir, 'test')

    write_ctf(args.datadir, 'train', train_data, vocab_dict)
    write_ctf(args.datadir, 'dev', valid_data, vocab_dict)
    write_ctf(args.datadir, 'test', test_data, vocab_dict)
