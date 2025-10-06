import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
import numpy as np
import aggregation
from client import Client, localModel
import argparse
from utils import setup_seed, start_attack, create_model
from load_data import load_and_separate_data

# np.warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", type=str, default='cifar10')
    parser.add_argument("--batchsize", type=int, help="batchsize", default=128)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.0001)
    parser.add_argument("--bias", help="way to assign data to workers", type=float, default=0.7)
    parser.add_argument("--gpu", help="index of gpu", type=int, default=0)
    parser.add_argument("--model", type=str, help="model", default='cnnc')
    parser.add_argument("--seed", type=int, help="random seed", default=1)
    parser.add_argument("--interval", type=int, help="log interval (epochs)", default=10)
    parser.add_argument("--epochs", type=int, help="number of epochs", default=600)
    parser.add_argument("--attack_epoch", type=int, help="the epoch that starts selfish attack", default=50)
    parser.add_argument("--local_round", help="number of local rounds", type=int, default=3)
    parser.add_argument("--nworkers", type=int, help="number of workers", default=20)
    parser.add_argument("--nbyz", type=int, help="number of Byzantine workers", default=6)
    parser.add_argument("--aggregation", type=str, help="'fedavg', 'median', 'trim'", default='fedavg')
    parser.add_argument("--alpha", type=float, help="alpha", default=0.0)
    parser.add_argument("--epsilon", type=float, help="parameter of starting attack", default=0.1)
    parser.add_argument("--cmax", type=int, help="parameter of trim", default=6)
    args = parser.parse_args()
    return args

def main(args):
    input_str = ' '.join(sys.argv)
    print(input_str)

    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device("cuda:" + str(args.gpu))

    setup_seed(args.seed)

    num_workers = args.nworkers
    local_round = args.local_round
    nbyz = args.nbyz
    alpha = args.alpha

    def evaluate_accuracy(dataLoader, net, device=torch.device('cpu')):
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for data in dataLoader:
                images_, labels_ = data
                images = images_.clone().to(device)
                labels = labels_.clone().to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = correct / total
        return test_acc

    paraString = str(args.seed) + "+" + str(args.dataset) + "+" + str(args.model) + "+" + "bs" + str(args.bias) + "+e" + str(args.epochs) + \
                "+" + "loc" + str(args.local_round) + "+lr" + str(args.lr) + "+" + "batch" + str(args.batchsize) + \
                "+" + "work" + str(args.nworkers) + "+" + "nbyz" + str(args.nbyz) + "+" + str(args.aggregation) + "+cmax" + str(args.cmax) + \
                "+attack_epoch" + str(args.attack_epoch) + "+alpha" + str(args.alpha) + "+epsilon" + str(args.epsilon)

    ###Load datasets
    each_worker_data, each_worker_label, test_data = load_and_separate_data(args, device)

    print([len(each_worker_data[i]) for i in range(num_workers)])

    Clients = []
    for each_worker in range(num_workers):
        model = create_model(args.model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4, eps=1e-6)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        clientModel = localModel(optimizer, model, criterion)
        Clients.append(Client(each_worker, clientModel))

    attack_epoch = float("inf")
    result = []
    epoch_interval = args.attack_epoch
    selfish_loss = []
    max_loss_gap = 0.0
    for each_epoch in range(args.epochs):
        test_acc_list = []
        params_list = []
        loss_list = []
        grad_list = []
        for each_worker in range(num_workers):
            grad, loss = Clients[each_worker].get_gradient(each_worker_data[each_worker], each_worker_label[each_worker],
                                                        local_round, args.batchsize, device)
            loss_list.append(loss)
            grad_list.append(grad)

        grad_list = torch.stack(grad_list, dim=0).squeeze(0)
        selfish_loss.append(np.mean(loss_list[:nbyz]))
        attack_epoch, max_loss_gap = start_attack(selfish_loss, max_loss_gap, each_epoch, attack_epoch, epoch_interval, args.epsilon)

        for each_worker in range(num_workers):
            Clients[each_worker].update_model(grad_list[each_worker])

        for each_worker in range(num_workers):
            params_list.append(Clients[each_worker].get_param())
        params_list = torch.stack(params_list, dim=0).squeeze(0)

        del grad_list

        for each_worker in range(num_workers):
            if args.aggregation == "fedavg":
                agg_para = aggregation.selfish_fedavg(params_list, nbyz, each_worker, each_epoch, attack_epoch, _lambda=alpha)
            elif args.aggregation == "median":
                agg_para = aggregation.selfish_median(params_list, nbyz, each_worker, each_epoch, attack_epoch, _lambda=alpha)
            elif args.aggregation == "trim":
                agg_para = aggregation.selfish_trim(params_list, nbyz, each_worker, each_epoch, attack_epoch, cmax=args.cmax, _lambda=alpha)
            else:
                raise NotImplementedError
            Clients[each_worker].set_param(agg_para)
            if (each_epoch % args.interval == 0 and each_epoch != 0) or each_epoch == args.epochs - 1:
                test_acc = evaluate_accuracy(test_data, Clients[each_worker].get_model(), device)
                test_acc_list.append(test_acc)

        if (each_epoch % args.interval == 0 and each_epoch != 0) or each_epoch == args.epochs - 1:
            max_acc = np.max(test_acc_list)
            min_acc = np.min(test_acc_list)
            median_acc = np.median(test_acc_list)
            mean_acc = np.mean(test_acc_list)
            gap_acc = np.mean(test_acc_list[:nbyz]) - np.mean(test_acc_list[nbyz:])
            print('[Epoch %d] maxAcc=%.4f, minAcc=%.4f, medianAcc=%.4f, meanAcc=%.4f, gapAcc=%.4f' % (
                each_epoch, max_acc, min_acc, median_acc, mean_acc, gap_acc))
            print(np.argsort(-np.array(test_acc_list)))
            print(-np.sort(-np.array(test_acc_list)))
            result.append([each_epoch, max_acc, min_acc, median_acc, mean_acc, np.mean(test_acc_list[:nbyz]),
                        np.mean(test_acc_list[nbyz:]), gap_acc])

        if each_epoch % 100 == 0:
            print(input_str)

    del Clients
    print(input_str)
    os.makedirs('result/', exist_ok=True)
    np.savetxt('result/' + paraString + '.txt', result)


if __name__ == "__main__":
    args = parse_args()
    main(args)
