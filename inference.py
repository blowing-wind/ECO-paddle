import numpy as np
import paddle.fluid as fluid
from models import TSN
from reader import KineticsReader

import argparse
parser = argparse.ArgumentParser(description="Paddle implementation of ECO")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'something'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--arch', type=str, default="ECOfull")
parser.add_argument('--load_path', type=str, default="")
parser.add_argument('--log_path', type=str, default="")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')

def main():
    args = parser.parse_args()
    if args.dataset == 'ucf101':
        args.num_class = 101
    elif args.dataset == 'hmdb51':
        args.num_class = 51
    elif args.dataset == 'kinetics':
        args.num_class = 400
    else:
        raise ValueError('Unknown dataset ' + args.dataset)
    
    place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        model = TSN(args.num_class, args.num_segments, args.modality, args.arch, dropout=0)
        args.short_size = model.scale_size
        args.target_size = model.crop_size
        args.input_mean = model.input_mean
        args.input_std = model.input_std * 3

        state_dict = fluid.dygraph.load_dygraph(args.load_path)[0]
        model.set_dict(state_dict)

        test_reader = KineticsReader('test', args, args.test_list).create_reader()
        log = open(args.log_path, 'w')

        model.eval()
        avg_acc = AverageMeter()

        for batch_id, data in enumerate(test_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int64')
                    
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            label.stop_gradient = True

            out, acc = model(img, label)

            avg_acc.update(acc.numpy()[0], label.shape[0])
            if (batch_id + 1) % args.print_freq == 0:
                output = 'Test batch_id:{} | acc {} | avg acc:{}'.format(
                    batch_id + 1, acc.numpy()[0], avg_acc.avg)
                print(output)
                log.write(output + '\n')
                log.flush()
        output = 'Test Avg acc:{}'.format(avg_acc.avg)
        print(output)
        log.write(output + '\n')
        log.flush()
        log.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
