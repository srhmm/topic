from operator import mod
from turtle import st
import numpy as np

np.set_printoptions(precision=4)

def polyfit(x, y, n_glob):
    n_loc = len(x)
    opt_cost = 0
    for i in range(1, 4):
        coeff, res, _, _, _ = np.polyfit(x, y, i, full=True)
        if res.size == 0:
            res = [0]
        res += 0.0001
        data_cost = np.log2(res/n_loc) * n_loc
        model_cost = (i+2)*np.log2(n_glob)
        cost = data_cost + model_cost
        if i == 1 or cost < opt_cost:
            opt_cost = cost
            opt_coeff = coeff
            opt_deg = i + 1
    
    return opt_cost, opt_deg, opt_coeff


# compute the scores for all possible bin merges, given the neighboring atomic starting bins
# scores[i,j] = Merge of i+1 bins starting from position j


def precomputeScores(binsx, binsy, fitting_fun):
    beta = len(binsx)
    if beta != len(binsy):
        raise ValueError(
            "X and y have different lengths {:d} and {:d}".format(beta, len(binsy)))
    scores = np.zeros((beta, beta))
    model_params = {}
    n = np.concatenate(binsx).shape[0]
    for i in range(beta):
        start = 0
        stop = start + i + 1
        while stop <= beta:
            data_split_x = np.concatenate(binsx[start:stop])
            data_split_y = np.concatenate(binsy[start:stop])
            #data_split_y = (data_split_y - np.min(data_split_y)) / (np.max(data_split_y) - np.min(data_split_y))
            if(np.isnan(np.sum(data_split_y))):
                scores[i,start] = 1000000
                start += 1
                stop += 1
                continue
            score_fit, deg, coeff = fitting_fun(data_split_x, data_split_y, n)
            model_params[(i, start)] = (deg, coeff)
            scores[i, start] = score_fit
            start += 1
            stop += 1
    return scores, beta, model_params


def HEC_Opt(binned_x, binned_y, fitting_fun=polyfit, show=False):
    precomputed_scores, num_bins, model_params = precomputeScores(
        binned_x, binned_y, fitting_fun)
    opt_cost = precomputed_scores.copy()

    opt_split = np.zeros((num_bins, num_bins, 4), dtype=np.int16)
    opt_split[0, :, :] = -1

    # bottom up approach with dynamic programming: for each bin merge we compute the optimal split
    # using the optimal subproblem structure until we arrive at the optimal split for the entire domain
    for i in range(1, num_bins):
        start = 0
        stop = start + i + 1
        while stop <= num_bins:
            # start with the no split split :)
            min_cost = opt_cost[i, start]
            opt_split[i, start, :] = -1
            # since we have already computed the optimal split costs for all splits size < i
            # only all binary split combinations have to be compared and the minimum cost found
            for j in range(i):
                split1cost = opt_cost[j, start]
                split2size = i - j - 1
                split2loc = start + j + 1
                split2cost = opt_cost[split2size, split2loc]
                new_cost = split1cost + split2cost
                if min_cost > new_cost:
                    min_cost = new_cost
                    opt_split[i, start, :] = [j, start, split2size, split2loc]
            opt_cost[i, start] = min_cost
            start += 1
            stop += 1
    final_cost = opt_cost[num_bins-1, 0]

    complete_model = aggregateModel(opt_split, model_params, num_bins-1, 0)
    #rescaled_score = modelPredict(complete_model, binned_x, binned_y, polyCompute)
    return final_cost,complete_model

def polyCompute(x, deg, coeff):
    pred = np.zeros(x.shape)
    for i in range(deg):
        pred += x**(deg-1-i) * coeff[i]
    return pred

def modelPredict(model, binsx, binsy, eval_fun):
    splits = model[0]
    params = model[1]
    n_global = len(np.concatenate(binsx))
    model_cost = 0
    for j in range(len(splits)):
        length, start = splits[j]
        deg, coeff = params[j]
        if j == len(splits)-1:
            end = len(binsx)
        else:
            end = start+length+1
        xbin = np.concatenate(binsx[start:end])
        n_local = len(xbin)
        pred = eval_fun(xbin, deg, coeff)
        ybin = np.concatenate(binsy[start:end])
        norm = np.max(ybin) - np.min(ybin)
        rss = np.sum(((pred - ybin)/norm)**2)
        model_cost += (deg + 1)*np.log2(n_global) + n_local*np.log2(rss/n_local)
    return model_cost

def aggregateModel(splitpoints, model_coefficients, cord1, cord2):
    cutpoint = splitpoints[cord1, cord2, :]
    if cutpoint[0] == -1:
        return [(cord1, cord2)], [model_coefficients[(cord1, cord2)]]
    else:
        split1cords, split1params = aggregateModel(
            splitpoints, model_coefficients, cutpoint[0], cutpoint[1])
        split2cords, split2params = aggregateModel(
            splitpoints, model_coefficients, cutpoint[2], cutpoint[3])
        return split1cords + split2cords, split1params + split2params


