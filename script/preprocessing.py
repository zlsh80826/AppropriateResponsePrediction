import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import jieba

invalid = ['', ' ', '\t']

def set_jieba(datadir, num_threads):
    jieba.set_dictionary(datadir + '/dictionary.txt')
    jieba.enable_parallel(num_threads)

def read_data(datadir):

    programs = []

    for i in range(1, 9):
        program = pd.read_csv(datadir + '/Program0%d.csv' % (i))
        programs.append(program)

    questions = pd.read_csv(datadir + '/Question.csv')

    return programs, questions

def tokenize(sentences, is_question):
    token = []
    
    for sentence in sentences:
        tokenized_sentence = jieba.lcut(sentence, cut_all=True)
        no_space = [t for t in tokenized_sentence if t not in invalid]
        if no_space:
            if is_question:
                token.extend(no_space)
            else:
                token.append(no_space)
    return token

def tokenize_programs(programs):
    tokenized_programs = []
    word2vec_train = []

    for program in tqdm(programs, desc='processing programs'):
        tokenized_program = []
    
        for index in range(len(program)):
            sentences = program.loc[index]['Content'].split('\n')
            tokens = tokenize(sentences, False)
            tokenized_program.append(tokens)
            word2vec_train.extend(tokens)
    
        tokenized_programs.append(tokenized_program)

    return tokenized_programs, word2vec_train

def tokenize_questions(questions):
    tokenized_questions = []
    n = len(questions)

    for i in tqdm(range(n), desc='processing questions'):
        tokenized_question = []
        sentences = questions.loc[i]['Question'].split('\n')
        tokenized_question.append(tokenize(sentences, True))
    
        for j in range(6):
            option = questions.loc[i]['Option%d' % (j)]
            tokenized_option = [t for t in jieba.lcut(option, cut_all=True) if t not in invalid]
            tokenized_question.append(tokenized_option)
    
        tokenized_questions.append(tokenized_question)

    return tokenized_questions

def sentence2context(programs):
    extended_programs = list()
    for program in tqdm(programs, desc='sentence to context'):
        extended_episode = list()
        for episode in tqdm(program):
            extended_sentence = list()
            zip_episodes = zip(episode[:-3], episode[1:-2], episode[2:-1])
            for s0, s1, s2 in zip_episodes:
                extended_sentence.append(s0 + s1 + s2)
            extended_episode.append(extended_sentence)
        extended_programs.append(extended_episode)
    return extended_programs

def save_npy(data_path, tps, tqs, w2v, eps):
    np.save(data_path + '/tokenized_programs.npy', tps)
    np.save(data_path + '/tokenized_questions.npy', tqs)
    np.save(data_path + '/word2vec_train.npy', w2v)
    np.save(data_path + '/extended_programs.npy', eps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Answer Prediction Preprocessing')
    parser.add_argument('--datadir', help='Data directory', required=False, default='../data')
    parser.add_argument('--threads', help='number of threads to jieba', required=False, default=1, type=int)
    args = parser.parse_args()

    set_jieba(args.datadir, args.threads)
    
    print('Start to tokenize the data, it may takes serveral minutes.')
    programs, questions = read_data(args.datadir)
    tps, w2v = tokenize_programs(programs)
    tqs = tokenize_questions(questions)
    eps = sentence2context(tps)

    save_npy(args.datadir, tps, tqs, w2v, eps)
