import os
import numpy as np
import paddle.fluid as fluid
from opts import parser
from models import TSN
from reader import KineticsReader

def main():
    place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        global args
        args = parser.parse_args()

        if args.dataset == 'ucf101':
            args.num_class = 101
        elif args.dataset == 'hmdb51':
            args.num_class = 51
        elif args.dataset == 'kinetics':
            args.num_class = 400
        else:
            raise ValueError('Unknown dataset '+args.dataset)
        
        model = TSN(args.num_class, args.num_segments, 
                    args.modality, args.arch, dropout=args.dropout)
        
        args.short_size = model.scale_size
        args.target_size = model.crop_size
        args.input_mean = model.input_mean
        args.input_std = model.input_std * 3

        if args.pretrained_parts == 'finetune':
            print('***Finetune model with {}***'.format(args.pretrained_model))

            state_dict = fluid.dygraph.load_dygraph(args.pretrained_model)[0]
            model_dict = model.state_dict()
            print('extra keys: {}'.format(set(list(state_dict.keys())) - set(list(model_dict.keys()))))
            print('missing keys: {}'.format(set(list(model_dict.keys())) - set(list(state_dict.keys()))))
            for k, v in state_dict.items():
                if 'fc' not in k:
                    model_dict.update({k:v})
            model.set_dict(model_dict)
        
        optimizer = fluid.optimizer.Momentum(args.lr, args.momentum, model.parameters(), 
        regularization=fluid.regularizer.L2Decay(args.weight_decay), 
        grad_clip=fluid.clip.GradientClipByGlobalNorm(args.clip_gradient))

        train_reader = KineticsReader('train', args, args.train_list).create_reader()
        val_reader = KineticsReader('val', args, args.val_list).create_reader()

        saturate_cnt = 0
        best_prec1 = 0
        log = open(os.path.join(args.log_path, args.save_name+'_train.csv'), 'w')

        for epoch in range(args.epochs):
            if saturate_cnt == args.num_saturate:
                print('learning rate decay by 0.1.')
                log.write('learning rate decay by 0.1. \n')
                log.flush()
                adjust_learing_rate(optimizer)
                saturate_cnt = 0
            train(train_reader, model, optimizer, epoch, log)

            if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
                prec1 = validate(val_reader, model, epoch, log)

                is_best = prec1 > best_prec1
                if is_best:
                    saturate_cnt = 0
                else:
                    saturate_cnt = saturate_cnt + 1
                    output = "- Validation Prec@1 saturates for {} epochs. Best acc{}".format(saturate_cnt, best_prec1)
                    print(output)
                    log.write(output + '\n')
                    log.flush()
                best_prec1 = max(prec1, best_prec1)

                if is_best:
                    fluid.dygraph.save_dygraph(model.state_dict(), os.path.join(args.save_dir, args.save_name))
        log.close()

def train(train_reader, model, optimizer, epoch, log):
    place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        model.train()
        epoch_acc = AverageMeter()
        epoch_loss = AverageMeter()

        for batch_id, data in enumerate(train_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int64')
                    
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            label.stop_gradient = True

            out, acc = model(img, label)

            loss = fluid.layers.cross_entropy(out, label)
            avg_loss = fluid.layers.mean(loss)

            avg_loss.backward()

            optimizer.minimize(avg_loss)
            model.clear_gradients()

            epoch_acc.update(acc.numpy()[0], label.shape[0])
            epoch_loss.update(avg_loss.numpy()[0], label.shape[0])
            if (batch_id + 1) % args.print_freq == 0:
                output = 'Epoch: {} batch_id:{} lr:{:.6f} Train Avg loss:{:.6f}, Avg acc:{:.6f}'.format(
                    epoch, batch_id + 1, optimizer.current_step_lr(), epoch_loss.avg, epoch_acc.avg)
                print(output)
                log.write(output + '\n')
                log.flush()
        
        output = 'Epoch: {} lr: {:.6f} Train Avg loss:{:.6f}, Avg acc:{:.6f}'.format(
            epoch, optimizer.current_step_lr(), epoch_loss.avg, epoch_acc.avg)
        print(output)
        log.write(output + '\n')
        log.flush()

def validate(val_reader, model, epoch, log):
    place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        model.eval()
        epoch_acc = AverageMeter()
        epoch_loss = AverageMeter()

        for batch_id, data in enumerate(val_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int64')
                    
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            label.stop_gradient = True

            out, acc = model(img, label)

            loss = fluid.layers.cross_entropy(out, label)
            avg_loss = fluid.layers.mean(loss)

            epoch_acc.update(acc.numpy()[0], label.shape[0])
            epoch_loss.update(avg_loss.numpy()[0], label.shape[0])
            if (batch_id + 1) % args.print_freq == 0:
                output = 'Epoch {}: batch_id:{} Validate Avg loss:{:.6f}, Avg acc:{:.6f}'.format(
                    epoch, batch_id + 1, epoch_loss.avg, epoch_acc.avg)
                print(output)
                log.write(output + '\n')
                log.flush()
        output = 'Epoch {} Validate Avg loss:{:.6f}, Avg acc:{:.6f}'.format(
            epoch, epoch_loss.avg, epoch_acc.avg)
        print(output)
        log.write(output + '\n')
        log.flush()

        return epoch_acc.avg

def adjust_learing_rate(optimizer):
    optimizer._learning_rate *= 0.1

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
