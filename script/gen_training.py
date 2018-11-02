import random
import numpy as np
import argparse
from tqdm import tqdm

def read_data(datadir):
    programs = np.load(datadir + '/extended_programs.npy')
    questions = np.load(datadir + '/tokenized_questions.npy')
    
    return programs, questions

def shuffle_context(programs):

    context = list()
    
    for program in programs:
        for episode in program:
            for sentence in episode:
                context.append(sentence)

    shuffle_list = list(range(len(context) - 1))
    random.shuffle(shuffle_list)

    return context, shuffle_list

def generate_testing_data(questions):
    x = []
    for sample in questions:
        question = sample[0]
        options = sample[1:]
        for option in options:
            x.append((question, option))
    return x

def write_question(datadir, x):
    with open(datadir + '/test.tsv', 'w', encoding='utf-8') as file:
        for xx in tqdm(x, desc='Writing test.tsv'):
            nt = [t for t in xx[0] if t != '\t']
            c1 = ' '.join(nt)
            nt = [t for t in xx[1] if t != '\t']
            c2 = ' '.join(nt)
            file.write(c1 + '\t' + c2 + '\n')

def generate_training_data(context, num_samples, shuffle_list):
    x, y = list(), list()
    count = 0
    for i in tqdm(range(num_samples), desc='Generating {} training data'.format(num_samples)):
        pos_or_neg = random.randint(0, 2)
        
        if pos_or_neg < 1:
            try:
                x.append((context[shuffle_list[count]], context[shuffle_list[count] + 3]))
            except:
                x.append((context[0], context[3]))
            y.append(1)
            count = (count + 1) % len(context)
            
        else:
            f = random.randint(0, len(context)-1)
            s = random.randint(0, len(context)-1)
            x.append((context[f], context[s]))
            y.append(0)
    
    return x, y

def write_tsv(datadir, type_, x, y):
    with open(datadir + '/{}.tsv'.format(type_), 'w', encoding='utf-8') as file:
        for xx, yy in tqdm(zip(x, y), desc='Writing {}.tsv'.format(type_)):
            nt = [t for t in xx[0] if t != '\t']
            c1 = ' '.join(nt)
            nt = [t for t in xx[1] if t != '\t']
            c2 = ' '.join(nt)
            file.write(c1 + '\t' + c2 + '\t' + str(yy) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data and write with tsv format')
    parser.add_argument('--datadir', help='Data directory', required=False, default='../data')
    parser.add_argument('--num_train', help='Number of training samples', required=False, default=6000000, type=int)
    parser.add_argument('--num_valid', help='Number of validate samples', required=False, default=10000, type=int)
    args = parser.parse_args()
    
    print('Reading data and shuffle ...')
    programs, questions = read_data(args.datadir)
    context, shuffle_list = shuffle_context(programs)

    train_x, train_y = generate_training_data(context, args.num_train, shuffle_list[:2000000])
    valid_x, valid_y = generate_training_data(context, args.num_valid, shuffle_list[2000000:])
    test = generate_testing_data(questions)

    write_tsv(args.datadir, 'train', train_x, train_y)
    write_tsv(args.datadir, 'valid', valid_x, valid_y)
    write_question(args.datadir, test)
