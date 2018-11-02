from model import Model
from helper import *
from config import *
from tqdm import tqdm
import cntk as C
import argparse

def train(ckpt, logdir):
    model = Model()
    z, loss, acc = model.model()
    
    progress_writers = [C.logging.ProgressPrinter(
                            num_epochs = max_epochs,
                            freq = log_freq,
                            tag = 'Training',
                            log_to_file = logdir + '/' + version + '.log')]
    
    lr = C.learning_parameter_schedule(learning_rate, minibatch_size=None, epoch_size=None)
    learner = C.adadelta(z.parameters, lr)
    trainer = C.Trainer(z, (loss, acc), learner, progress_writers)
    
    mb_source, input_map = deserialize(loss, train_data, model)
    mb_valid, valid_map = deserialize(loss, valid_data, model)
    
    try:
        trainer.restore_from_checkpoint(ckpt)
    except Exception:
        print('No checkpoint founded.')
    
    for epoch in range(max_epochs):
        num_seq = 0
        # train
        with tqdm(total=epoch_size, desc='Epoch {} train'.format(epoch)) as progress_bar:
            while True:
                data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
                trainer.train_minibatch(data)
                num_seq += trainer.previous_minibatch_sample_count
                progress_bar.update(trainer.previous_minibatch_sample_count)
                if num_seq >= epoch_size:
                    break
            trainer.summarize_training_progress()
            trainer.save_checkpoint('../model/' + version + '/' + str(epoch)) 
        num_seq = 0        
        # validation
        with tqdm(total=num_validation, desc='Epoch {} validate'.format(epoch)) as valid_progress_bar:
            while True:
                data = mb_valid.next_minibatch(minibatch_size, input_map=valid_map)
                if not data:
                    break
                trainer.test_minibatch(data)
                num_seq += len(data)
                valid_progress_bar.update(len(data))
                if num_seq >= num_validation:
                    break
            trainer.summarize_test_progress()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--ckpt', help='Restart from checkpoint, give the checkpoint realpath', required=False, default=None)
    parser.add_argument('--logdir', help='Log directory', required=False, default='../log')
    args = parser.parse_args()

    train(args.ckpt, args.logdir)

    
