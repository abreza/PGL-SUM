# -*- coding: utf-8 -*-
from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    """ Main function that sets the data loaders; trains and evaluates the model."""
    config = get_config()
    test_config = get_config(mode='test')

    print(config)
    print(test_config)
    print('Currently selected split_index:', config.split_index)

    pretrain_loader = None
    if config.n_pretrain_epochs:
        pretrain_loader = get_loader('train', 'activitynet', pretrain=True)
    train_loader = get_loader(config.mode, config.video_type, config.split_index)
    test_loader = get_loader(test_config.mode, test_config.video_type, test_config.split_index)

    solver = Solver(config, pretrain_loader, train_loader, test_loader)

    solver.build()
    if config.zero_shot:
        solver.evaluate(-1, from_pretrain=config.from_pretrain)	 # evaluates the summaries using the initial random weights of the network
    if config.n_pretrain_epochs:
        solver.pretrain()
    if config.mode == 'train':
        solver.train(from_pretrain=config.from_pretrain)

# tensorboard --logdir '../PGL-SUM/Summaries/PGL-SUM/'
