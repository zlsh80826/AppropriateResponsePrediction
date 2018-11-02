import warnings
warnings.filterwarnings('ignore')
from config import *
from helper import *
from model import Model
import cntk as C
import numpy as np
from tqdm import tqdm
import argparse

def inference(model, test):
    p = Model()
    model = C.load_model(model)

    cos = model.outputs[0]
    loss = C.as_composite(model.outputs[1].owner)

    mb_test, map_test = deserialize(loss, test, p, 
                                    randomize=False, repeat=False, is_test=True)
    c1 = argument_by_name(loss, 'c1')
    c2 = argument_by_name(loss, 'c2')

    results = []
    if 'test' in test:
        total_samples = 3000
    else:
        total_samples = num_validation
        
    with tqdm(total=total_samples) as progress_bar:
        while True:
            data = mb_test.next_minibatch(minibatch_size, input_map=map_test)
            progress_bar.update(len(data))
            if not data:
                break
            out = model.eval(data, outputs=[cos])
            results.extend(out)
    assert(len(results) == total_samples)
    return results

def write_prediction(results, datadir, output):
    questions = np.load(datadir + '/tokenized_questions.npy')
    predict = list()
    for index in range(0, len(results), 6):
        sample = results[index:index + 6]
        predict.append(np.argmax(sample))

    with open(output, 'w', encoding='utf-8') as out:
        out.write('Id,Answer\n')
        for idx, p in enumerate(predict):
            out.write(str(idx) + ',' + str(p) + '\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference model')
    parser.add_argument('--model', help='Model weights file', required=False, default='../model/' + version + '/0')
    parser.add_argument('--test', help='Test file in ctf format', required=False, default='../data/test.ctf')
    parser.add_argument('--answer', help='Answer file name', required=False, default=version + '_answer.csv')
    parser.add_argument('--datadir', help='Data directory', required=False, default='../data')
    args = parser.parse_args()

    results = inference(args.model, args.test)
    write_prediction(results, args.datadir, args.answer)
    print('Answer has been written to', args.answer)
