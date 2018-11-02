import numpy as np
import argparse
from gensim.models import FastText

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train embedding with fasttext')
    parser.add_argument('--datadir', help='Data directory', required=False, default='../data')
    parser.add_argument('--embeddingdir', help='Directory to store the embedding weights', required=False, default='../embedding')
    parser.add_argument('--dim', help='embedding dimensions', required=False, default=300, type=int)
    parser.add_argument('--ws', help='windows size', required=False, default=12, type=int)
    parser.add_argument('--min_count', help='minium count for rare words', required=False, default=5, type=int)
    parser.add_argument('--threads', help='number of threads to use for training', required=False, default=4, type=int)
    parser.add_argument('--iter', help='number of iterations to train', required=False, default=50, type=int)
    args = parser.parse_args()

    sentences = np.load(args.datadir + '/word2vec_train.npy')
    model = FastText(sentences, size=args.dim, window=args.ws, min_count=args.min_count, 
                                workers=args.threads, iter=args.iter)
    model.save(args.embeddingdir + '/fasttext.bin')
