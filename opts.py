import argparse
parser = argparse.ArgumentParser(description="Paddle implementation of ECO")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'something'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)
parser.add_argument('--pretrained_model', type=str, default=None)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="ECOfull")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
parser.add_argument('--pretrained_parts', type=str, default='both',
                    choices=['scratch', '2D', '3D', 'both','finetune'])
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-i', '--iter-size', default=1, type=int,
                    metavar='N', help='number of iterations before on update')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--boundaries', default=[596], type=float, nargs="+",
                    metavar='LRSteps', help='iters to decay learning rate')
parser.add_argument('--values', default=[0.001, 0.0001], type=float, nargs="+",
                    metavar='LRSteps', help='learning rates')                    
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
parser.add_argument('--nesterov',  default=False)
parser.add_argument('--num_saturate', type=int, default=5,
                    help='if number of epochs that validation Prec@1 saturates, then decrease lr by 10 (default: 5)')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save_dir', type=str, default="")
parser.add_argument('--log_path', type=str, default="")
parser.add_argument('--save_name', type=str, default="ECO_full_model")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--rgb_prefix', default="", type=str)







