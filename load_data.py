import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def load_and_separate_data(args, device):
    if args.dataset == 'cifar10':
        batch_size = args.batchsize
        num_workers = args.nworkers
        num_outputs = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download='True', transform=transform_train)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download='True', transform=transform_test)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        del trainset, testset
        bias_weight = args.bias
        other_group_size = (1 - bias_weight) / (num_outputs - 1)
        worker_per_group = num_workers / num_outputs
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]
        for _, (data, label) in enumerate(train_data):
            for (x, y) in zip(data, label):
                upper_bound = (y.item()) * (1 - bias_weight) / (num_outputs - 1) + bias_weight
                lower_bound = (y.item()) * (1 - bias_weight) / (num_outputs - 1)
                rd = np.random.random_sample()
                if rd > upper_bound:
                    worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.item() + 1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.item()

                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                if args.bias == 0: selected_worker = np.random.randint(num_workers)
                each_worker_data[selected_worker].append(x.to(device))
                each_worker_label[selected_worker].append(y.to(device))

        # concatenate the data for each worker
        each_worker_data = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_data]
        each_worker_label = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_label]
        # random shuffle the workers
        # random_order = np.random.RandomState(seed=args.seed).permutation(num_workers)
        # each_worker_data = [each_worker_data[i] for i in random_order]
        # each_worker_label = [each_worker_label[i] for i in random_order]
        order = [10, 11, 15, 12, 17, 5, 6, 7, 8, 9, 0, 1, 3, 4, 14, 2, 16, 13, 18, 19]
        each_worker_data = [each_worker_data[i] for i in order]
        each_worker_label = [each_worker_label[i] for i in order]
        return each_worker_data, each_worker_label, test_data

    else:
        raise NotImplementedError