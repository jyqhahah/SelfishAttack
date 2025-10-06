import torch

def selfish_fedavg(param_list, nbyz, each_worker, each_epoch, attack_epoch, pure=False, _lambda=0.0):
    if each_epoch < attack_epoch:
        agg_para = torch.mean(param_list, dim=0)
    elif each_worker < nbyz:  # selfish client
        if pure:
            agg_para = torch.mean(param_list[:nbyz], dim=0)
        else:
            agg_para = torch.mean(param_list, dim=0)
    else:
        mean_arr = torch.mean(param_list[nbyz:], dim=0)
        if _lambda == 1.0:
            sorted_array, _ = torch.sort(param_list[nbyz:], dim=0, descending=True)
            local_model = mean_arr.clone()
            local_model[param_list[each_worker] > mean_arr] = sorted_array[0][param_list[each_worker] > mean_arr]
            local_model[param_list[each_worker] < mean_arr] = sorted_array[-1][param_list[each_worker] < mean_arr]
        elif _lambda < 1.0:
            sorted_array, _ = torch.sort(param_list[nbyz:], dim=0, descending=True)
            local_model = (param_list[each_worker] - mean_arr * _lambda) / (1 - _lambda)
            compared_model = local_model.clone()
            local_model[compared_model > sorted_array[0]] = sorted_array[0][compared_model > sorted_array[0]]
            local_model[compared_model < sorted_array[-1]] = sorted_array[-1][compared_model < sorted_array[-1]]
        else:
            sorted_array, _ = torch.sort(param_list[nbyz:], dim=0, descending=True)
            local_model = (param_list[each_worker] - mean_arr * _lambda) / (1 - _lambda)
            high_arr = sorted_array[0]
            low_arr = sorted_array[-1]
            high_dist = torch.abs(local_model - high_arr)
            low_dist = torch.abs(local_model - low_arr)
            local_model[high_dist > low_dist] = sorted_array[0][high_dist > low_dist]
            local_model[high_dist <= low_dist] = sorted_array[-1][high_dist <= low_dist]
        agg_para = local_model
    return agg_para

def selfish_median(param_list, nbyz, each_worker, each_epoch, attack_epoch, pure=False, _lambda=0.0):
    if each_epoch < attack_epoch:  # selfish client
        sorted_array = torch.sort(param_list, dim=0)
        if len(param_list) % 2 == 1:
            agg_para = sorted_array[0][int(len(param_list) / 2), :]
        else:
            agg_para = (sorted_array[0][int(len(param_list) / 2) - 1, :] + sorted_array[0][
                                                                           int(len(param_list) / 2), :]) / 2
        del sorted_array
    elif each_worker < nbyz:
        if pure:
            agg_para = torch.mean(param_list[:nbyz], dim=0)
        else:
            sorted_array = torch.sort(param_list, dim=0)
            if len(param_list) % 2 == 1:
                agg_para = sorted_array[0][int(len(param_list) / 2), :]
            else:
                agg_para = (sorted_array[0][int(len(param_list) / 2) - 1, :] + sorted_array[0][
                                                                                int(len(param_list) / 2), :]) / 2
            del sorted_array
    else:
        sorted_array, sorted_index = torch.sort(param_list[nbyz:], dim=0)
        if len(param_list[nbyz:]) % 2 == 1:
            median_arr = sorted_array[int(len(sorted_array) / 2), :]
        else:
            median_arr = (sorted_array[int(len(sorted_array) / 2) - 1, :] + sorted_array[
                                                                            int(len(sorted_array) / 2), :]) / 2
        if _lambda == 1.0:
            local_model = median_arr.clone()
            local_model[param_list[each_worker] > median_arr] = sorted_array[0][param_list[each_worker] > median_arr]
            local_model[param_list[each_worker] < median_arr] = sorted_array[-1][param_list[each_worker] < median_arr]
        elif _lambda < 1.0:
            local_model = (param_list[each_worker] - median_arr * _lambda) / (1 - _lambda)
        else:
            local_model = (param_list[each_worker] - median_arr * _lambda) / (1 - _lambda)
            if len(param_list) % 2 == 1:
                high_arr = sorted_array[(len(sorted_array) - nbyz - 1) // 2]
                low_arr = sorted_array[(len(sorted_array) + nbyz - 1) // 2]
            else:
                high_arr = (sorted_array[(len(sorted_array) - nbyz) // 2] + sorted_array[(len(sorted_array) - nbyz) // 2 - 1]) / 2
                low_arr = (sorted_array[(len(sorted_array) + nbyz) // 2] + sorted_array[(len(sorted_array) + nbyz) // 2] - 1) / 2
            high_dist = torch.abs(local_model - high_arr)
            low_dist = torch.abs(local_model - low_arr)
            local_model[high_dist > low_dist] = sorted_array[0][high_dist > low_dist]
            local_model[high_dist <= low_dist] = sorted_array[-1][high_dist <= low_dist]
        attack_high = sorted_array[(len(sorted_array)-nbyz) // 2]
        attack_low = sorted_array[(len(sorted_array)+nbyz-1) // 2]
        attack_model = local_model.clone()
        attack_model[local_model > attack_high] = (2*local_model-attack_high)[local_model > attack_high]
        attack_model[local_model < attack_low] = (2*local_model-attack_low)[local_model < attack_low]
        del sorted_array
        attacked_model = torch.stack([attack_model] * nbyz, dim=0).to(median_arr.device)

        sorted_array = param_list.clone()
        sorted_array[:nbyz] = attacked_model
        del attacked_model
        sorted_array = torch.sort(sorted_array, dim=0)
        if len(param_list) % 2 == 1:
            agg_para = sorted_array[0][int(len(param_list) / 2), :]
        else:
            agg_para = (sorted_array[0][int(len(param_list) / 2) - 1, :] + sorted_array[0][
                                                                           int(len(param_list) / 2), :]) / 2
        del sorted_array
    return agg_para

def selfish_trim(param_list, nbyz, each_worker, each_epoch, attack_epoch, pure=False, cmax=6, _lambda=0.0):
    if each_epoch < attack_epoch:  # selfish client
        sorted_array = torch.sort(param_list, dim=0)
        agg_para = torch.mean(sorted_array[0][cmax:len(param_list) - cmax, :], dim=0)
        del sorted_array
    elif each_worker < nbyz:
        if pure:
            agg_para = torch.mean(param_list[:nbyz], dim=0)
        else:
            sorted_array = torch.sort(param_list, dim=0)
            agg_para = torch.mean(sorted_array[0][cmax:len(param_list) - cmax, :], dim=0)
            del sorted_array

    else:
        sorted_array, _ = torch.sort(param_list[nbyz:], dim=0, descending=True)
        mean_arr = torch.mean(sorted_array[cmax:len(sorted_array) - cmax], dim=0)
        sum_arr = torch.sum(sorted_array[cmax:len(sorted_array) - cmax], dim=0)
        start_high = cmax - 1
        start_low = len(sorted_array) - cmax
        prefix_high = [sum_arr]
        prefix_low = [sum_arr]
        if _lambda == 1.0:
            local_model = mean_arr.clone()
            local_model[param_list[each_worker] > mean_arr] = sorted_array[0][param_list[each_worker] > mean_arr]
            local_model[param_list[each_worker] < mean_arr] = sorted_array[-1][param_list[each_worker] < mean_arr]
        elif _lambda < 1.0:
            local_model = (param_list[each_worker] - mean_arr * _lambda) / (1 - _lambda)
        else:
            local_model = (param_list[each_worker] - mean_arr * _lambda) / (1 - _lambda)
            high_arr = torch.mean(sorted_array[:len(sorted_array)-cmax], dim=0)
            low_arr = torch.mean(sorted_array[cmax:], dim=0)
            high_dist = torch.abs(local_model - high_arr)
            low_dist = torch.abs(local_model - low_arr)
            local_model[high_dist > low_dist] = sorted_array[0][high_dist > low_dist]
            local_model[high_dist <= low_dist] = sorted_array[-1][high_dist <= low_dist]
        for i in range(start_high + 1):
            prefix_high.append(prefix_high[-1] + sorted_array[start_high - i])
            prefix_low.append(prefix_low[-1] + sorted_array[start_low + i])  # len = start_high+2 = cmax+1
        attacked_model = ((len(sorted_array) - cmax) * local_model - sum_arr) / cmax
        attacked_models_low = torch.stack([attacked_model] * nbyz, dim=0).to(local_model.device)
        attacked_models_high = torch.stack([attacked_model] * nbyz, dim=0).to(local_model.device)
        for i in range(len(prefix_low)):  # k = start_low + i - 1
            k = start_low + i
            tmp_low = ((len(sorted_array) - k) * sorted_array[cmax - 1] + prefix_low[i]) / (len(sorted_array) - cmax)
            if k >= len(sorted_array) - cmax:
                for j in range(nbyz):
                    if j < len(sorted_array) - k:
                        attacked_low = ((len(sorted_array) - cmax) * local_model - prefix_low[i]) / (
                                len(sorted_array) - k)
                        attacked_models_low[j][local_model <= tmp_low] = attacked_low[local_model <= tmp_low]
                    else:
                        attacked_low = sorted_array[-1] - 1.0
                        attacked_models_low[j][local_model <= tmp_low] = attacked_low[local_model <= tmp_low]
        for i in range(len(prefix_high)):
            k = start_high - i
            tmp_high = ((k + 1) * sorted_array[len(sorted_array) - cmax] + prefix_high[i]) / (len(sorted_array) - cmax)
            if k <= cmax - 1:
                for j in range(nbyz):
                    if j < k + 1:
                        attacked_high = ((len(sorted_array) - cmax) * local_model - prefix_high[i]) / (k + 1)
                        attacked_models_high[j][local_model >= tmp_high] = attacked_high[local_model >= tmp_high]
                    else:
                        attacked_high = sorted_array[0] + 1.0
                        attacked_models_high[j][local_model >= tmp_high] = attacked_high[local_model >= tmp_high]
        del sorted_array
        attacked_models = torch.stack([local_model] * nbyz, dim=0).to(local_model.device)
        attacked_models[:, local_model <= mean_arr] = attacked_models_low[:, local_model <= mean_arr]
        attacked_models[:, local_model > mean_arr] = attacked_models_high[:, local_model > mean_arr]
        del attacked_models_low, attacked_models_high
        sorted_array = param_list.clone()
        sorted_array[:nbyz] = attacked_models
        del attacked_models
        sorted_array = torch.sort(sorted_array, dim=0)
        agg_para = torch.mean(sorted_array[0][cmax:len(param_list) - cmax, :], dim=0)
        del sorted_array
    return agg_para
