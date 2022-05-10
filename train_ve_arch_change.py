import sys
import os.path
import argparse
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data_ve
import model_arch_change_counting as model_arch_change
import utils

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
cnt_iter_train = 0
cnt_iter_val = 0
best_train_acc = None
best_val_acc = None


def run(net, loader, optimizer, scheduler, tracker, train=False, has_answers=True, prefix='', epoch=0):
    global writer, cnt_iter_train, cnt_iter_val, best_train_acc, best_val_acc
    """ Run an epoch over the given loader """
    assert not (train and not has_answers)
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {
            'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(
        prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(
        prefix), tracker_class(**tracker_params))
    for v, q, a, b, idx, q_len in loader:
        var_params = {
            'volatile': not train,
            'requires_grad': False,
        }
        v = Variable(v.cuda(non_blocking=True), **var_params)

        # UNCOMMENT FOR LSTM
        # q = Variable(q.cuda(non_blocking=True), **var_params)

        a = Variable(a.cuda(non_blocking=True), **var_params)
        b = Variable(b.cuda(non_blocking=True), **var_params)
        q_len = Variable(q_len.cuda(non_blocking=True), **var_params)

        out = net(v, b, q, q_len)
        if has_answers:
            # nll = -F.log_softmax(out, dim=1)
            # loss = (nll * a / 10).sum(dim=1).mean()
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(out, a)
            acc = utils.batch_accuracy(out.data, a.data).cpu()

        if train:
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            if has_answers:
                accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        if has_answers:
            loss_tracker.append(loss.item())
            if train:
                writer.add_scalar("train/loss", loss.item(), cnt_iter_train)
            else:
                writer.add_scalar("val/loss", loss.item(), cnt_iter_val)
            acc_tracker.append(acc.mean())
            if train:
                acc_log = acc.mean().item()
                writer.add_scalar(
                    "train/acc", acc_log, cnt_iter_train)
                if best_train_acc is None or acc_log > best_train_acc:
                    best_train_acc = acc_log
                    torch.save(net.state_dict(), "model_best_train_acc.pt")
            else:
                acc_log = acc.mean().item()
                writer.add_scalar("val/acc", acc_log, cnt_iter_val)
                if best_val_acc is None or acc_log > best_val_acc:
                    best_val_acc = acc_log
                    torch.save(net.state_dict(), "model_best_val_acc.pt")
            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(loss_tracker.mean.value),
                               acc=fmt(acc_tracker.mean.value))
        if train:
            cnt_iter_train += 1
        else:
            cnt_iter_val += 1

    if not train:
        answ = list(torch.cat(answ, dim=0))
        if has_answers:
            accs = list(torch.cat(accs, dim=0))
        else:
            accs = []
        idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='*')
    parser.add_argument('--eval', dest='eval_only', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', nargs='*')
    args = parser.parse_args()
    print(args)
    # return
    if args.test:
        args.eval_only = True
    src = open('model_arch_change_counting.py').read()
    if args.name:
        name = ' '.join(args.name)
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    if not args.test:
        # target_name won't be used in test mode
        print('will save to {}'.format(target_name))
    if args.resume:
        logs = torch.load(' '.join(args.resume))
        # hacky way to tell the VQA classes that they should use the vocab without passing more params around
        data.preloaded_vocab = logs['vocab']

    cudnn.benchmark = True

    if not args.eval_only:
        # train_loader = data.get_loader(train=True)
        train_loader = data_ve.get_ve_loader(mode="train")
    if not args.test:
        val_loader = data_ve.get_ve_loader(mode="val")
    # else:
    #     # Saumya - Not written code for this yet!
    #     val_loader = data.get_loader(test=True)

    net = model_arch_change.Net(val_loader.dataset.num_tokens).cuda()

    # VISUALIZATION
    # net.load_state_dict(torch.load(
    #     "analysis_counting/models/transformer_naive_bbox_count_module/counting/model_best_val_acc.pt"))
    # v = torch.unsqueeze(train_loader.dataset[58408][0], dim=0).cuda()
    # q = [train_loader.dataset[58408][1]]
    # # a = torch.unsqueeze(train_loader.dataset[58408][2], dim=0).cuda()
    # b = torch.unsqueeze(train_loader.dataset[58408][3], dim=0).cuda()
    # q_len = torch.tensor(train_loader.dataset[58408][5]).cuda()
    # tmp_output = net(v, b, q, q_len)

    optimizer = optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=config.initial_lr)
    scheduler = lr_scheduler.ExponentialLR(
        optimizer, 0.5**(1 / config.lr_halflife))
    if args.resume:
        net.load_state_dict(logs['weights'])

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(
        config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        print("About to start epoch number ", i)
        if not args.eval_only:
            print("Beginning training")
            run(net, train_loader, optimizer, scheduler,
                tracker, train=True, prefix='train', epoch=i)

        print("Beginning validation")
        r = run(net, val_loader, optimizer, scheduler, tracker,
                train=False, prefix='val', epoch=i, has_answers=not args.test)

        if not args.test:
            results = {
                'name': name,
                'tracker': tracker.to_dict(),
                'config': config_as_dict,
                'weights': net.state_dict(),
                'eval': {
                    'answers': r[0],
                    'accuracies': r[1],
                    'idx': r[2],
                },
                'vocab': val_loader.dataset.vocab,
                'src': src,
            }
            torch.save(results, target_name)
        else:
            # in test mode, save a results file in the format accepted by the submission server
            answer_index_to_string = {
                a:  s for s, a in val_loader.dataset.answer_to_index.items()}
            results = []
            for answer, index in zip(r[0], r[2]):
                answer = answer_index_to_string[answer.item()]
                qid = val_loader.dataset.question_ids[index]
                entry = {
                    'question_id': qid,
                    'answer': answer,
                }
                results.append(entry)
            with open('results.json', 'w') as fd:
                json.dump(results, fd)

        if args.eval_only:
            break


if __name__ == '__main__':
    main()
