import torch
import numpy as np


class Client:
    def __init__(self, client_id, model):
        self.client_id = client_id
        self.model = model

    def get_requires_grad_idx(self):
        return self.model.get_requires_grad_idx()

    def get_model(self):
        return self.model.get_net()

    def get_client_id(self):
        return self.client_id

    def train(self, data, label, local_round, batch_size, device):
        return self.model.train(data, label, local_round, batch_size, device)

    def get_param(self):
        return self.model.get_param()

    def set_param(self, agg_para):
        self.model.set_param(agg_para)

    def get_gradient(self, data_, label_, local_round, batch_size, device):
        return self.model.compute_gradient(data_, label_, local_round, batch_size, device)

    def update_model(self, grad):
        self.model.update_model(grad)

    def get_attacked_model(self, grad):
        return self.model.get_attacked_model(grad)


class localModel:
    def __init__(self, optimizer, net, criterion, schedular=None):
        self.optimizer = optimizer
        self.net = net
        self.criterion = criterion
        self.schedular = schedular
        self.requires_grad_list = [name for name, param in self.net.named_parameters() if param.requires_grad != False]

    def get_requires_grad_idx(self):
        res_list = []
        for name, param in self.net.state_dict().items():
            if name in self.requires_grad_list:
                res_list.append(torch.zeros(param.shape))
            elif 'running' in name:
                res_list.append(torch.ones(param.shape))
        return torch.cat([xx.reshape((-1)) for xx in res_list], dim=0).squeeze(0).bool()

    def train(self, data_, label_, local_round, batch_size, device):
        loss_num = 0.0
        for _ in range(local_round):
            minibatch = np.random.choice(list(range(data_.shape[0])), size=batch_size, replace=False)
            data = data_[minibatch].to(device)
            label = label_[minibatch].to(device)
            self.net.train()
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = self.criterion(output, label)
            loss_num = loss.item()
            loss.backward()
            self.optimizer.step()
            if self.schedular is not None:
                self.schedular.step()
        return loss_num

    def get_param(self):
        tmp_params = [param.clone() for param in self.net.parameters() if param.requires_grad != False]
        return torch.cat([xx.reshape((-1)) for xx in tmp_params], dim=0).squeeze(0)

    def set_param(self, agg_para):
        idx = 0
        for j, (param) in enumerate(self.net.named_parameters()):
            if param[1].requires_grad:
                tmp_param = agg_para[idx:(idx + param[1].nelement())].reshape(param[1].shape)
                param[1].data = tmp_param
                idx += param[1].nelement()

    def get_net(self):
        return self.net

    def compute_gradient(self, data_, label_, local_round, batch_size, device):
        old_params = self.get_param()
        loss_num = 0.0
        cnt = 0
        for k in range(local_round):
            random_order = np.random.permutation(len(data_))
            for j in range((len(data_) - 1) // batch_size + 1):
                data = data_[random_order[j * batch_size: min(len(data_), (j + 1) * batch_size)]].to(device)
                label = label_[random_order[j * batch_size: min(len(data_), (j + 1) * batch_size)]].to(device)
                self.net.train()
                self.optimizer.zero_grad()
                output = self.net(data)
                loss = self.criterion(output, label)
                loss_num += loss.item()
                cnt += 1
                loss.backward()
                self.optimizer.step()
                if self.schedular is not None:
                    self.schedular.step()
        grad = self.get_param() - old_params
        self.set_param(old_params)
        return grad, loss_num / cnt

    def update_model(self, grad):
        idx = 0
        for param in self.net.parameters():
            if param.requires_grad:
                param.data += grad[idx:(idx + param.nelement())].reshape(param.shape)  
                idx = idx + param.nelement()

    def get_attacked_model(self, grad):
        prev_model = self.get_param()
        return prev_model + grad
