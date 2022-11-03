import math
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from copy import deepcopy
from datetime import timedelta
from collections import OrderedDict
from timeit import default_timer as timer

from utils.logger import Logger
from utils.data import LabeledDocuments
from utils.torch_helper import move_to_device, squeeze_dim
from utils.evaluation import compute_retrieval_precision


class Base_Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.load_data()

    def load_data(self):
        self.data = LabeledDocuments(self.hparams)

    def get_hparams_grid(self):
        raise NotImplementedError

    def define_parameters(self):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def configure_gradient_clippers(self):
        raise NotImplementedError

    def run_training_sessions(self):
        logger = Logger(self.hparams.model_path + '.log', on=True)
        val_perfs = []
        best_val_perf = float('-inf')
        start = timer()
        random.seed(self.hparams.seed)  # For reproducible random runs

        for run_num in range(1, self.hparams.num_runs + 1):
            state_dict, val_perf = self.run_training_session(run_num, logger)
            val_perfs.append(val_perf)

            if val_perf > best_val_perf:
                best_val_perf = val_perf
                logger.log('----New best {:8.2f}, saving'.format(val_perf))
                torch.save({'hparams': self.hparams,
                            'state_dict': state_dict}, self.hparams.model_path)

        logger.log('Time: %s' % str(timedelta(seconds=round(timer() - start))))
        self.load()
        if self.hparams.num_runs > 1:
            logger.log_perfs(val_perfs)
            logger.log('best hparams: ' + self.flag_hparams())

        val_perf, test_perf = self.run_test()
        logger.log('Val:  {:8.2f}'.format(val_perf))
        logger.log('Test: {:8.2f}'.format(test_perf))


    def run_training_session(self, run_num, logger):
        self.train()

        # Scramble hyperparameters if number of runs is greater than 1.
        if self.hparams.num_runs > 1:
            logger.log('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
            for hparam, values in self.get_hparams_grid().items():
                assert hasattr(self.hparams, hparam)
                self.hparams.__dict__[hparam] = random.choice(values)

        random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)

        self.define_parameters()
        # logger.log(str(self))
        logger.log('%d params' % sum([p.numel() for p in self.parameters()]))
        logger.log('hparams: %s' % self.flag_hparams())

        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        self.to(device)

        optimizer = self.configure_optimizers()
        gradient_clippers = self.configure_gradient_clippers()
        train_loader, database_loader, val_loader, _ = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=True, get_test=False)
        best_val_perf = float('-inf')
        best_state_dict = None
        bad_epochs = 0

        # logger.log('Time: %s' % str(timedelta(seconds=round(timer() - start))))
        try:

            for epoch in range(1, self.hparams.epochs + 1):
                epoch_start = timer()

                forward_sum = {}
                num_steps = 0
                for batch_num, batch in enumerate(train_loader):
                    optimizer.zero_grad()

                    inputs = squeeze_dim(
                        move_to_device(batch[0], device), dim=1)
                    forward = self.forward(inputs)

                    for key in forward:
                        if key in forward_sum:
                            forward_sum[key] += forward[key]
                        else:
                            forward_sum[key] = forward[key]
                    num_steps += 1

                    if math.isnan(forward_sum['loss']):
                        logger.log('Stopping epoch because loss is NaN')
                        break

                    forward['loss'].backward()
                    for params, clip in gradient_clippers:
                        nn.utils.clip_grad_norm_(params, clip)
                    optimizer.step()

                if math.isnan(forward_sum['loss']):
                    logger.log('Stopping training session because loss is NaN')
                    break

                val_perf = self.evaluate(
                    database_loader, val_loader, device, is_val=True)
                logger.log('End of epoch {:3d}'.format(epoch), False)
                logger.log(' '.join([' | {:s} {:8.2f}'.format(
                    key, forward_sum[key] / num_steps)
                    for key in forward_sum]), False)
                logger.log(' | val perf {:8.2f}'.format(val_perf), False)
                logger.log(' | time : %s' %
                           str(timedelta(seconds=round(timer() - epoch_start))))

                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    bad_epochs = 0
                    logger.log('\t *Best model so far, deep copying*')
                    best_state_dict = deepcopy(self.state_dict())
                else:
                    bad_epochs += 1
                    logger.log('\t Bad epoch %d' % bad_epochs)

                if bad_epochs > self.hparams.num_bad_epochs:
                    break
        except KeyboardInterrupt:
            logger.log('-' * 89)
            logger.log('Exiting from training early')

        return best_state_dict, best_val_perf

    def evaluate(self, database_loader, eval_loader, device, is_val=True):
        self.eval()
        with torch.no_grad():
            perf = compute_retrieval_precision(database_loader, eval_loader,
                                               device, self.encode_continuous,
                                               self.pq_head.C,
                                               self.hparams.N_books,
                                               self.hparams.num_retrieve)
        self.train()
        return perf

    def run_test(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        _, database_loader, val_loader, test_loader = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=True, get_test=True)
        val_perf = self.evaluate(
            database_loader, val_loader, device, is_val=True)
        test_perf = self.evaluate(
            database_loader, test_loader, device, is_val=False)
        return val_perf, test_perf

    def load(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        checkpoint = torch.load(self.hparams.model_path) if self.hparams.cuda \
            else torch.load(self.hparams.model_path,
                            map_location=torch.device('cpu'))
        if checkpoint['hparams'].cuda and not self.hparams.cuda:
            checkpoint['hparams'].cuda = False
        self.hparams = checkpoint['hparams']
        self.define_parameters()
        self.load_state_dict(checkpoint['state_dict'])
        self.to(device)

    def flag_hparams(self):
        flags = '%s %s' % (self.hparams.model_path, self.hparams.data_path)
        for hparam in vars(self.hparams):
            val = getattr(self.hparams, hparam)
            if str(val) == 'False':
                continue
            elif str(val) == 'True':
                flags += ' --%s' % (hparam)
            elif str(hparam) in {'model_path', 'data_path', 'num_runs',
                                 'num_workers'}:
                continue
            else:
                flags += ' --%s %s' % (hparam, val)
        return flags

    @staticmethod
    def get_general_hparams_grid():
        grid = OrderedDict({
        })
        return grid


    @staticmethod
    def get_general_argparser():
        parser = argparse.ArgumentParser()

        parser.add_argument('model_path', type=str)
        parser.add_argument('data_path', type=str)
        parser.add_argument('--train', action='store_true',
                            help='train a model?')

        parser.add_argument('--num_runs', type=int, default=1,
                            help='num random runs (not random if 1) '
                            '[%(default)d]')
        parser.add_argument('--seed', type=int, default=123,
                            help='random seed [%(default)d]')
        parser.add_argument("--batch_size", default=128, type=int,
                            help='batch size [%(default)d]')
        parser.add_argument('--epochs', type=int, default=80,
                            help='max number of epochs [%(default)d]')
        parser.add_argument("--lr", default=1e-4, type=float,
                            help='initial learning rate [%(default)g]')
        # parser.add_argument('--bert_lr', default = 5e-5, type = float,
        #                     help='bert initial learning rate [%(default)g]')
        parser.add_argument("-l", "--encode_length", type=int, default=32,
                            help="Number of bits of the code [%(default)d]")

        parser.add_argument('--cuda', action='store_true',
                            help='use CUDA?')
        parser.add_argument('--num_workers', type=int, default=8,
                            help='num dataloader workers [%(default)d]')
        parser.add_argument('--max_length', type=int, default=200,
                            help='max number of sentence length [%(default)d]')
        parser.add_argument('--num_retrieve', type=int, default=100,
                            help='num neighbors to retrieve [%(default)d]')
        parser.add_argument('--num_bad_epochs', type=int, default=8,
                            help='num indulged bad epochs [%(default)d]')
        parser.add_argument('--clip', type=float, default=10,
                            help='gradient clipping [%(default)g]')

        return parser
