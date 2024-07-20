import sys
import os
import tempfile
import argparse
import logging
from functools import partial
import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from ray import train, tune
from ray.train import RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

from mllp.utils import read_csv, DBEncoder
from mllp.models_l0 import L0MLLP


DATA_DIR = 'dataset'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def plot_loss(args, loss_log, accuracy, accuracy_b, f1_score, f1_score_b):
    set_name = 'validation' if args.use_validation_set else 'training'

    fig = plt.figure(figsize=(16, 16))
    fig.suptitle('Dataset: {}'.format(args.data_set), fontsize=16)
    plt.subplot(3, 1, 1)
    loss_array = np.array(loss_log)

    plt.plot(loss_array, color='b', label='Total loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss during the training')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(np.array(accuracy), color='b', label='MLLP')
    plt.plot(np.array(accuracy_b), color='g', label='CRS')

    plt.xlabel('epoch * 5')
    plt.ylabel('Accuracy')
    plt.title('Accuracy on the {} set'.format(set_name))
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(np.array(f1_score), color='b', label='MLLP')
    plt.plot(np.array(f1_score_b), color='g', label='CRS')

    plt.xlabel('epoch * 5')
    plt.ylabel('F1 Score Micro')
    plt.title('F1 Score (Macro) on the {} set'.format(set_name))
    plt.grid(True)
    plt.legend()

    plt.savefig(args.plot_file)
  
def experiment(args, data_path, info_path):
    args = argparse.Namespace(**args) # included to convert back the dict required for the tuner to the args object

    wandb = setup_wandb(vars(args), rank_zero_only=False, project='l0_{}_k{}_ki{}_useValidationSet{}_e{}_bs{}_useNOT{}_N{}_local_rep{}_group_l0{}'.format(args.data_set, args.kfold, args.ith_kfold, args.use_validation_set, args.epoch, args.batch_size, args.use_not, args.N, args.local_rep, args.group_l0))

    # Create temp dir
    tempdirname = tempfile.TemporaryDirectory().name

    args.folder_name = "artifacts"

    # args.folder_name = 'l0_{}_k{}_ki{}_useValidationSet{}_e{}_bs{}_lr{}_lrdr{}_lrde{}_wd{}_useNOT{}_lamba{}_droprate_init_input{}_droprate_init{}_N{}_beta_ema{}_local_rep{}_temperature{}_group_l0{}'.format(
    #     args.data_set, args.kfold, args.ith_kfold, args.use_validation_set, args.epoch, args.batch_size,
    #     args.learning_rate, args.lr_decay_rate, args.lr_decay_epoch, args.weight_decay, args.use_not, args.lamba, args.droprate_init_input, args.droprate_init, args.N, args.beta_ema, args.local_rep, args.temperature, args.group_l0)

    if not os.path.exists(os.path.join(tempdirname, 'log_folder')):
        os.makedirs(os.path.join(tempdirname, 'log_folder'))
    # args.folder_name = args.folder_name + '_L' + args.structure
    args.folder_path = os.path.join(tempdirname, 'log_folder', args.folder_name)
    if not os.path.exists(args.folder_path):
        os.makedirs(args.folder_path)
    args.model = os.path.join(args.folder_path, 'model.pth')
    args.crs_file = os.path.join(args.folder_path, 'crs.txt')
    args.plot_file = os.path.join(args.folder_path, 'plot_file.pdf')
    args.log = os.path.join(args.folder_path, 'log.txt')
    logging.basicConfig(level=logging.INFO, filename=args.log, filemode='w', format='[%(levelname)s] - %(message)s')

    # dataset = args.data_set

    # data_path = os.path.join(DATA_DIR, dataset + '.data')
    # info_path = os.path.join(DATA_DIR, dataset + '.info')

    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=0)
    train_index, test_index = list(kf.split(X_df))[args.ith_kfold]
    X_train_df = X_df.iloc[train_index]
    y_train_df = y_df.iloc[train_index]
    X_test_df = X_df.iloc[test_index]
    y_test_df = y_df.iloc[test_index]

    logging.info('Discretizing and binarizing data. Please wait ...')
    db_enc = DBEncoder(f_df, discrete=True)
    db_enc.fit(X_df, y_df)
    X_fname = db_enc.X_fname
    y_fname = db_enc.y_fname
    X_train, y_train = db_enc.transform(X_train_df, y_train_df)
    X_test, y_test = db_enc.transform(X_test_df, y_test_df)
    logging.info('Data discretization and binarization are done.')

    if args.use_validation_set:
        # Use 20% of the training set as the validation set.
        # kf = KFold(n_splits=5, shuffle=True, random_state=0) # In the original implementation this is hard-coded and could be missaligned with the above kfold split?
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=0)
        train_index, validation_index = next(kf.split(X_train))
        X_validation = X_train[validation_index]
        y_validation = y_train[validation_index]
        X_train = X_train[train_index]
        y_train = y_train[train_index]
    else:
        X_validation = None
        y_validation = None

    net_structure = [len(X_fname)] + list(map(int, args.structure.split('_'))) + [len(y_fname)]
    net = L0MLLP(net_structure,
               device=device,
               use_not=args.use_not,
               log_file=None,
               N=args.N if args.N is not None else len(X_train_df),
               beta_ema=args.beta_ema,
               weight_decay=args.weight_decay,
               lamba=args.lamba,
               droprate_init_input=args.droprate_init_input,
               droprate_init=args.droprate_init,
               local_rep=args.local_rep,
               temperature=args.temperature,
               group_l0=args.group_l0)
    net.to(device)

    loss_log, accuracy, accuracy_b, f1_score, f1_score_b = net.train(
        X_train,
        y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay)

    plot_loss(args, loss_log, accuracy, accuracy_b, f1_score, f1_score_b)

    acc, acc_b, f1, f1_b = net.test(X_test, y_test, need_transform=True)
    logging.info('=' * 60)
    logging.info('Test:\n\tAccuracy of MLLP Model: {}\n\tAccuracy of CRS  Model: {}'.format(acc, acc_b))
    logging.info('Test:\n\tF1 Score of MLLP Model: {}\n\tF1 Score of CRS  Model: {}'.format(f1, f1_b))
    logging.info('=' * 60)

    with open(args.crs_file, 'w') as f:
        net.concept_rule_set_print(X_train, X_fname, y_fname, f)
    torch.save(net.state_dict(), args.model)

    # Send the logs to Tune (and it will upload them to Wandb)
    for i in range(0, args.epoch - 1):
        report_dict = {"epoch": i}
        report_dict.update({"loss": loss_log[i]})
        if i % 5 == 0:
            report_dict.update({"train_accuracy": accuracy[int(i / 5)], "train_accuracy_b": accuracy_b[int(i / 5)], "train_f1_score": f1_score[int(i / 5)], "train_f1_score_b": f1_score_b[int(i / 5)]})
        wandb.log(report_dict)
        train.report(report_dict)
    report_dict = {"epoch": args.epoch - 1}
    report_dict.update({"loss": loss_log[args.epoch - 1]})
    if (args.epoch - 1) % 5 == 0:
        report_dict.update({"train_accuracy": accuracy[args.epoch - 1], "train_accuracy_b": accuracy_b[args.epoch - 1], "train_f1_score": f1_score[args.epoch - 1], "train_f1_score_b": f1_score_b[args.epoch - 1]})
    report_dict.update({"test_accuracy": acc, "test_accuracy_b": acc_b, "test_f1_score": f1, "test_f1_score_b": f1_b})
    wandb.log(report_dict)
    wandb.log_artifact(args.folder_path)
    wandb.finish()
    train.report(report_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Arguments that will be passed or defaulted
    parser.add_argument('-d', '--data_set', type=str, default='connect-4',
                        help='Set the data set for training. All the data sets in the dataset folder are available.')
    parser.add_argument('-k', '--kfold', type=int, default=5, help='Set the k of K-Folds cross-validation.')
    parser.add_argument('-ki', '--ith_kfold', type=int, default=0, help='Do the i-th validation, 0 <= ki < k.')
    parser.add_argument('--use_validation_set', action="store_true",
                        help='Use the validation set for parameters tuning.', default=True)
    parser.add_argument('-e', '--epoch', type=int, default=401, help='Set the total epoch.')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Set the batch size.')
    parser.add_argument('-ns', '--num_samples', type=int, help='Number of samples for the hyperparameter tuning search')
    parser.add_argument('-N', type=int, default=None,
                        help='L0 N parameter')
    parser.add_argument('--use_not', action="store_true",
                        help='Use the NOT (~) operator in logical rules. '
                             'It will enhance model capability but make the CRS more complex.')
    parser.add_argument('--local_rep', action="store_true",
                        help='L0 local_rep parameter')
    parser.add_argument('--group_l0', action="store_true",
                        help='L0 group_l0 parameter')

    # Arguments that will be passed or set up by the tuner
    parser.add_argument('-lr', '--learning_rate', type=float, default=None, help='Set the initial learning rate.')
    parser.add_argument('-lrdr', '--lr_decay_rate', type=float, default=None, help='Set the learning rate decay rate.')
    parser.add_argument('-lrde', '--lr_decay_epoch', type=int, default=None, help='Set the learning rate decay epoch.')
    parser.add_argument('-wd', '--weight_decay', type=float, default=None, help='Set the weight decay (L2 penalty).')
    parser.add_argument('--lamba', type=float, default=None,#1,
                        help='L0 Lamba parameter')
    parser.add_argument('--droprate_init_input', type=float, default=None,
                        help='L0 droprate_init_input parameter')
    parser.add_argument('--droprate_init', type=float, default=None,
                        help='L0 droprate_init parameter')
    parser.add_argument('--beta_ema', type=float, default=None,
                        help='L0 beta_ema parameter')
    parser.add_argument('--temperature', type=float, default=None,
                        help='L0 temperature parameter')
    parser.add_argument('-s', '--structure', type=str, default=None, # '64,
                        help='Set the structure of network. Only the number of nodes in middle layers are needed. '
                             'E.g., 64, 64_32_16. The total number of middle layers should be odd.')

    # Set logging
    # logging.basicConfig(level=logging.ERROR, stream=sys.stdout, format='[%(levelname)s] - %(message)s')

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    
    args = parser.parse_args()

    config = {
        "data_set": args.data_set,
        "kfold": args.kfold,
        "ith_kfold": args.ith_kfold,
        "use_validation_set": args.use_validation_set,
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "structure": args.structure if args.structure is not None else tune.choice(["32", "64", "128", "256", "32_32_32", "64_64_64", "128_128_128", "256_256_256"]),
        "N": args.N,
        "use_not": args.use_not,
        "learning_rate": args.learning_rate if args.learning_rate is not None else tune.qloguniform(1e-4, 1e-1, 5e-5),
        "lr_decay_rate": args.lr_decay_rate if args.lr_decay_rate is not None else tune.quniform(0.1, 1.0, 0.05),
        "lr_decay_epoch": args.lr_decay_epoch if args.lr_decay_epoch is not None else tune.randint(0, args.epoch - 1),
        "weight_decay": args.weight_decay if args.weight_decay is not None else tune.quniform(0.0, 0.1, 5e-5),
        "lamba": args.lamba if args.lamba is not None else tune.qloguniform(1e-4, 1e-1, 5e-5),
        "droprate_init_input": args.droprate_init_input if args.droprate_init_input is not None else tune.quniform(0.05, 1.0, 0.05),
        "droprate_init": args.droprate_init if args.droprate_init is not None else tune.quniform(0.05, 1.0, 0.05),
        "beta_ema": args.beta_ema if args.beta_ema is not None else tune.quniform(0.05, 1.0, 0.05),
        "local_rep": args.local_rep,
        "temperature": args.temperature if args.temperature is not None else tune.quniform(0.05, 1.0, 0.05),
        "group_l0": args.group_l0
    }

    data_path = os.path.join(os.path.join(os.path.dirname(__file__), DATA_DIR), args.data_set + '.data')
    info_path = os.path.join(os.path.join(os.path.dirname(__file__), DATA_DIR), args.data_set + '.info')

    tuner = tune.Tuner(partial(experiment, data_path=data_path, info_path=info_path),
                    tune_config=tune.TuneConfig(
                                num_samples=args.num_samples
                                ),
                    # run_config=RunConfig(
                    #             callbacks=[WandbLoggerCallback(project='l0_{}_k{}_ki{}_useValidationSet{}_e{}_bs{}_useNOT{}_N{}_local_rep{}_group_l0{}'.format(args.data_set, args.kfold, args.ith_kfold, args.use_validation_set, args.epoch, args.batch_size, args.use_not, args.N, args.local_rep, args.group_l0))]
                    #             ),
                    param_space=config)

    results = tuner.fit()

    # args.folder_name = 'l0_{}_k{}_ki{}_useValidationSet{}_e{}_bs{}_lr{}_lrdr{}_lrde{}_wd{}_useNOT{}_lamba{}_droprate_init_input{}_droprate_init{}_N{}_beta_ema{}_local_rep{}_temperature{}'.format(
    #     args.data_set, args.kfold, args.ith_kfold, args.use_validation_set, args.epoch, args.batch_size,
    #     args.learning_rate, args.lr_decay_rate, args.lr_decay_epoch, args.weight_decay, args.use_not, args.lamba, args.droprate_init_input, args.droprate_init, args.N, args.beta_ema, args.local_rep, args.temperature)

    # if not os.path.exists('log_folder'):
    #     os.mkdir('log_folder')
    # args.folder_name = args.folder_name + '_L' + args.structure
    # args.folder_path = os.path.join('log_folder', args.folder_name)
    # if not os.path.exists(args.folder_path):
    #     os.mkdir(args.folder_path)
    # args.model = os.path.join(args.folder_path, 'model.pth')
    # args.crs_file = os.path.join(args.folder_path, 'crs.txt')
    # args.plot_file = os.path.join(args.folder_path, 'plot_file.pdf')
    # args.log = os.path.join(args.folder_path, 'log.txt')
    # logging.basicConfig(level=logging.INFO, filename=args.log, filemode='w', format='[%(levelname)s] - %(message)s')
    # for arg in vars(args):
    #     print(arg, getattr(args, arg))
    # experiment(args)
