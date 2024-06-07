from timeshap.explainer.kernel import TimeShapKernel
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from scipy.linalg import cholesky, cho_solve
from scipy.special import comb
import itertools
import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from copy import deepcopy
from itertools import combinations
import matplotlib as mpl
import matplotlib.pyplot as plt

class ConsecutiveMobiusKernel(TimeShapKernel):
    def __init__(self, model, background, rs, output_path=None):
        super().__init__(model, background, rs, "event")
        self.output_path = output_path
    
    def solve(self, fraction_evaluated, dim):

        self.n_terms = np.sum(np.arange(self.M, 0, -1)) 

        X = np.zeros((self.maskMatrix.shape[0]+1, self.n_terms), dtype='float32')
        X[:-1, :], terms = self._find_consecutive_interactions(self.maskMatrix)
        X[-1, :] = 1

        y = np.hstack([self.linkfv(self.ey[:, dim]) - self.link.f(self.fnull[dim]), self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])])
        W = np.hstack([np.ones(self.maskMatrix.shape[0]), np.array([1e6])]) 

        # regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_seed) # works fine
        # regressor = Lasso(alpha=1, fit_intercept=True, random_state=self.random_seed) # Lasso makes all mabius transforms 0
        regressor = LinearRegression(fit_intercept=True) # works fine
        regressor.fit(X, y, sample_weight=W)
        m = regressor.coef_
        m0 = regressor.intercept_

        phi = np.zeros(self.M)
        m_list = []
        for term, mobius in zip(terms, m):
            events = term.split('.')
            events[0] = events[0][1:]
            for event in events:
                phi[int(event)-1] += mobius/len(events)

            m_list += [[term, mobius]]

        self.mobius_transforms = dict(m_list)
        self.m0 = m0
        
        if self.output_path:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

            pd.DataFrame(self.mobius_transforms).to_csv(f"{self.output_path}/mobius_transforms.csv", index=False)
            pd.DataFrame(phi, columns=['Shapley Value']).to_csv(f"{self.output_path}/shapley_values.csv", index=False)
 
        return phi, np.ones(len(phi))

    def _find_consecutive_interactions(self, X):
        
        terms = [f"m{str(int(i))}" for i in np.arange(1, self.M+1)]

        for order in range(1, self.M):
            consecutives = np.zeros((X.shape[0], self.M - order)) 
            for j in range(self.M - order):
                name = ['m'] +  ['.'.join([str(int(j)+1) for j in np.arange(j, j+order+1)])]
                terms += [''.join(name)]

                consecutives[:, j] = np.prod(X[:, j:j+order+1], axis=1)
            X = np.concatenate((X, consecutives), axis=1)

        return X, terms

class CompleteMobiusKernel(TimeShapKernel):
    def __init__(self, model, background, rs, mode, output_path=None, lam=0.1):
        super().__init__(model, background, rs, mode)
        self.lam = lam

    def solve(self, fraction_evaluated, dim):

        eyAdj = self.linkfv(self.ey[:, dim]) - self.link.f(self.fnull[dim])
        mat = self.maskMatrix.copy()
        mat = np.append(mat, np.array([np.ones((self.M)), np.zeros((self.M))]), axis=0)
        inner_prod = mat @ mat.T

        sample_weights = np.append(np.ones((self.nsamples,)), [100000000]*2, axis=0)
        Omega = 2**inner_prod - 1
        y_hat = np.append(eyAdj, (self.fx[dim] - self.link.f(self.fnull[dim]), self.link.f(self.fnull[dim])))

        sample_set_size = np.array(np.sum(mat, 1), dtype=int) 
        size_weight = np.zeros((self.M,))
        for i in range(1,self.M+1):
            for j in range(1,i+1):
                size_weight[i-1] += (1/j) * comb(i-1,j-1)

        alpha_weight = np.array([size_weight[t-1] if t != 0 else  0 for t in sample_set_size])

        lam = self.lam  #0.1 
        L = cholesky(Omega + lam * np.diag(sample_weights) , lower=True)
        alpha = cho_solve((L, True), y_hat)

        shapley_val = np.zeros((self.M,))

        for i in range(self.M):
            #shapley_val[i] = (alpha_weight_sv * X_sv[:,i]) @ alpha
            shapley_val[i] = (alpha_weight * mat[:,i]) @ alpha

        return shapley_val, np.ones(len(shapley_val))

def run_experiment(f, data, baseline, pruning_idx, show_plot=True, output_path=None):
    # iterate over nsamples
    features = data.shape[1] - pruning_idx + 1
    max_features = features if 2**features <= 2**13 else 13
    nsamples_list = [2**f for f in range(5, max_features+1, 1)]

    explainer = TimeShapKernel(f, baseline, 42, "event")
    true = explainer.shap_values(data, nsamples=2**features, pruning_idx=pruning_idx)

    kernels = [TimeShapKernel, ConsecutiveMobiusKernel, CompleteMobiusKernel] #TODO: switch fast back to normal
    results = {kernel.__name__: {'error samples': [[]], 'median error': [], 'std error': [], 'runtime samples': [[]], 'median runtime': [], 'std runtime': []} for kernel in kernels}

    for nsamples in tqdm.tqdm(nsamples_list):

        for _ in range(5):
            rs = np.random.randint(0, 1000)

            for kernel in kernels:
                start = time.time()
                explainer = kernel(f, baseline, rs)
                phi = explainer.shap_values(data, nsamples=nsamples, pruning_idx=pruning_idx)
                end = time.time()

                mse = mean_squared_error(true, phi)
                results[kernel.__name__]['error samples'][-1] += [mse]
                results[kernel.__name__]['runtime samples'][-1] += [end-start]

        for kernel in kernels:
            results[kernel.__name__]['median error'] += [np.median(results[kernel.__name__]['error samples'][-1])]
            results[kernel.__name__]['std error'] += [np.std(results[kernel.__name__]['error samples'][-1])]
            results[kernel.__name__]['error samples'] += [[]]

            results[kernel.__name__]['median runtime'] += [np.median(results[kernel.__name__]['runtime samples'])]
            results[kernel.__name__]['std runtime'] += [np.std(results[kernel.__name__]['runtime samples'])]
            results[kernel.__name__]['runtime samples'] = [[]]


    fig_error, error = plt.subplots(figsize=(10, 5))
    colors = ['blue', 'red', 'green']
    for kernel, color in zip(kernels, colors):
        error.errorbar(nsamples_list, results[kernel.__name__]['median error'], label=kernel.__name__, fmt='-o', yerr=results[kernel.__name__]['std error'], color=color)

    error.set_xlabel('Number of Samples')
    error.set_xscale('log')
    error.set_xticks(nsamples_list)
    error.set_ylabel('Error')
    error.set_title(f'Comparison of Error for Different Kernels (n={features} features)')
    error.legend()
    error.grid(True)
    if show_plot: plt.show()

    fig_runtime, runtime = plt.subplots(figsize=(10, 5))
    for kernel, color in zip(kernels, colors):
        runtime.errorbar(nsamples_list, results[kernel.__name__]['median runtime'], label=kernel.__name__, fmt='-o', yerr= results[kernel.__name__]['std runtime'], color=color)

    runtime.set_xlabel('Number of Samples')
    runtime.set_xscale('log')
    runtime.set_xticks(nsamples_list)
    runtime.set_ylabel('Runtime')
    runtime.set_title(f'Comparison of Runtime for Different Kernels (n={features} features)')
    runtime.legend()
    runtime.grid(True)
    if show_plot: plt.show()

    if output_path:
        output_path += '/' + str(features)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            fig_error.savefig(os.path.join(output_path, 'error.png'))
            fig_runtime.savefig(os.path.join(output_path, 'runtime.png'))
        else:
            print('output path already exists, please remove it first.')

    return results


class HEDGE:
    def __init__(self, model, data, baseline=None, pruning_idx=0, win_size=2):
        self.pred = model(data)[0][0]
        self.pred_label = round(self.pred)
   
        self.pruning_idx = pruning_idx
        self.fea_num = data.shape[1] - self.pruning_idx

        self.model = model
        self.data = data
        self.win_size = win_size 

        if baseline is None:
            self.baseline = np.expand_dims(np.zeros(data.shape[1]), axis=[0,2])
        else:
            self.baseline = baseline
        self.bias = model(baseline)[0][0]
        self.valdict = dict()

    def set_contribution_func(self, model, fea_set, inputs):
        data = self.baseline.copy()

        for fea in fea_set:
            if type(fea) == int:
                data[0, fea, 0] = self.data[0, fea, 0]
            else:
                for f in fea:
                    data[0, f, 0] = self.data[0, f, 0]
        score = model(data)[0][0]

        return score - self.bias 

    def get_shapley_interaction_weight(self, d, s):
        return np.math.factorial(s) * np.math.factorial(d - s - 2) / np.math.factorial(d - 1) / 2
   
    def shapley_interaction_score_approx(self, model, input, feature_set, left, right, win_size):
        if left + 1 != right:
            print("Not adjacent interaction")
            return -1
        fea_num = len(feature_set)
        curr_set_lr = list((feature_set[left], feature_set[right]))
        curr_set_l = [feature_set[left]] if type(feature_set[left]) == int else feature_set[left]
        curr_set_r = [feature_set[right]] if type(feature_set[right]) == int else feature_set[right]
        if left + 1 - win_size > 0: # is the left set within bounds
            left_set = feature_set[left - win_size:left]
        else:
            left_set = feature_set[0:left]
        if right + win_size > fea_num - 1:
            right_set = feature_set[right + 1:]
        else:
            right_set = feature_set[right + 1:right + win_size + 1]
        adj_set = left_set + right_set
        num_adj = len(adj_set)
        dict_subset = {r: list(combinations(adj_set, r)) for r in range(num_adj+1)}
        for i in range(num_adj+1):
            weight = self.get_shapley_interaction_weight(fea_num, i)
            if i == 0:
                score_included = self.set_contribution_func(model, curr_set_lr, input)
                score_excluded_l = self.set_contribution_func(model, curr_set_r, input)
                score_excluded_r = self.set_contribution_func(model, curr_set_l, input)
                score_excluded = self.set_contribution_func(model, [], input)
                score = (score_included - score_excluded_l - score_excluded_r + score_excluded) * weight
            else:
                for subsets in dict_subset[i]:
                    score_included = self.set_contribution_func(model, list(subsets) + curr_set_lr, input)
                    score_excluded_l = self.set_contribution_func(model, list(subsets) + curr_set_r, input)
                    score_excluded_r = self.set_contribution_func(model, list(subsets) + curr_set_l, input)
                    score_excluded = self.set_contribution_func(model, list(subsets), input)
                    score += (score_included - score_excluded_l - score_excluded_r + score_excluded) * weight
        return score

    def shapley_topdown_tree(self):
        model = self.model
        data = self.data
        fea_num = self.fea_num
        fea_set = [list(range(fea_num))]
       
        self.hier_tree = {}
        self.hier_tree[0] = [(fea_set[0], self.set_contribution_func(model, fea_set[0], data))]

        for level in range(1, self.fea_num):
            pos = 0
            min_inter_score = 1e8
            while pos < len(fea_set):
                subset = fea_set[pos]
                sen_len = len(subset)
                if sen_len == 1:
                    pos += 1
                    continue
                new_fea_set = [ele for x, ele in enumerate(fea_set) if x != pos]
                score_buff = []
                for idx in range(1, sen_len):
                    leave_one_set = deepcopy(new_fea_set)
                    sub_set1 = subset[0:idx]
                    sub_set2 = subset[idx:]
                    leave_one_set.insert(pos, sub_set1)
                    leave_one_set.insert(pos + 1, sub_set2)
                    score_buff.append(self.shapley_interaction_score_approx(model, data, leave_one_set, pos, pos + 1, self.win_size))
                inter_score = np.array(score_buff)
                min_inter_idx = np.argmin(inter_score)
                minter = inter_score[min_inter_idx]
                if minter < min_inter_score:
                    min_inter_score = minter
                    inter_idx_opt = min_inter_idx
                    pos_opt = pos
                pos += 1

            new_fea_set = [ele for x, ele in enumerate(fea_set) if x != pos_opt]
            subset = fea_set[pos_opt]
            sub_set1 = subset[0:inter_idx_opt + 1]
            sub_set2 = subset[inter_idx_opt + 1:]
            new_fea_set.insert(pos_opt, sub_set1)
            new_fea_set.insert(pos_opt + 1, sub_set2)
            fea_set = new_fea_set
            new_level = []
            for fea in fea_set:
                new_level.append((fea, self.set_contribution_func(model, fea, data)))
            self.hier_tree[level] = new_level
        self.max_level = level
        self.shap_values = [fea[1] for fea in self.hier_tree[self.max_level]][::-1] # these are not actually shap values but make it easier for for evaluation to name it like this, reverse to compy with timeshap architecture
        return self.hier_tree

    def visualize_tree(self, wordvocab, fontsize=8, folder='model/now', tag='x'):
        levels = self.max_level

        cnorm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)
        cmapper = mpl.cm.ScalarMappable(norm=cnorm, cmap='RdYlBu_r')

        words = self.data[0, self.pruning_idx:, 0]
        nwords = len(words)
        fig, ax = plt.subplots(figsize=(nwords, nwords/2))
        ax.xaxis.set_visible(False)

        ylabels = ['Step '+str(idx) for idx in range(self.max_level+1)] + ['Baseline']
        ax.set_yticks(list(range(0, self.max_level+2)))
        ax.set_yticklabels(ylabels,fontsize=18)
        ax.set_ylim(self.max_level+1.5, 0-0.5)
        
        sep_len = 0.3
        for level in range(levels+1):
            for fea in self.hier_tree[level]:
                len_fea = 1 if type(fea[0]) == int else len(fea[0])
                start_fea = fea[0] if type(fea[0])==int else fea[0][0]	
                start = sep_len * start_fea + start_fea + 0.5
                width = len_fea + sep_len * (len_fea - 1)
                fea_color = cmapper.to_rgba(fea[1])
                ax.barh(level, width=width, height=0.5, left=start, color=fea_color)
                text_color = 'black'
                ax.text(start + width / 2, level-0.2, str(round(fea[1], 2)), ha='center', va='center', color=text_color, fontsize=fontsize)
                word_idxs = fea[0]
                for i, idx in enumerate(word_idxs):
                    word_pos = start + sep_len * (i) + i + 0.5
                    word_str = wordvocab[words[idx]]
                    ax.text(word_pos, level, word_str, ha='center', va='center',
                            color=text_color, fontsize=fontsize)
                    word_pos += sep_len
                start += (width + sep_len)
        # add baseline
        height = levels+1
        width = len(words) + sep_len * (len(words) - 1)
        start = 0.5
        value = self.bias
        if hasattr(self, 'm0'): value += self.m0
        fea_color = cmapper.to_rgba(value)
        ax.barh(height, width=width, height=0.5, left=start, color=fea_color)
        ax.text(start + width / 2, height, str(round(value, 2)), ha='center', va='center', color=text_color, fontsize=12)
        cb = fig.colorbar(cmapper, ax=ax)
        cb.ax.tick_params(labelsize=18)
        ax.set_xlim(0, nwords + nwords*sep_len + 0.5)
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f'{folder}/visualization_sentence_{tag}.png')
        # plt.show()
    
    
    def get_importance_phrase(self, num=-1):
        hier_list = []

        for level in range(1, self.max_level + 1):
            for fea_set, score in self.hier_tree[level]:
                hier_list.append((fea_set, score))
        hier_list = sorted(hier_list, key=lambda item: abs(item[1]), reverse=True)
        phrase_list = []
        if num == -1:
            num = 10000
        pre_items = []
        score_list = []
        count = 0
        for items, score in hier_list:
            if count == num:
                break
            if not set(items) == set(pre_items):
                phrase_list.append(items)
                score_list.append(score)
                pre_items = items
                count += 1
        return phrase_list, score_list




class MobiusHEDGE(HEDGE): # note, events in mobius are counted in opposite direction from HEDGE (1 is last event in mobius, 0 is first event in HEDGE) 
    def __init__(self, model, data, baseline=None, pruning_idx=0, win_size=None):
        super().__init__(model, data, baseline=baseline, pruning_idx=pruning_idx, win_size=win_size)
        explainer = ConsecutiveMobiusKernel(model, baseline, 42)
        self.shap_values = explainer.shap_values(data, nsamples=256, pruning_idx=pruning_idx) 
        self.m = explainer.mobius_transforms
        self.m0 = explainer.m0
        print('mobius transform calculated')

    def v(self, features):
        'value wrt bias of a span of features calculated with mobius transform'

        features = sorted([-(f-self.fea_num) for f in features])

        value = 0
        for i, start_feature in enumerate(features):
            name = 'm' + str(start_feature)
            value += self.m[name]

            for next_feature in features[i+1:]:
                name += '.' + str(next_feature)
                value += self.m[name]

        return value

    def mobius_interaction(self, feature_set, left, right):
        'Calculates the value of a coalition if features can only interact within their span.'

        value = self.v(feature_set[left]) + self.v(feature_set[right])

        prior_level = list(self.hier_tree.keys())[-1]
        prior_value = self.hier_tree[prior_level][left][1]

        return abs(prior_value - value)

    def find_highest_interaction(self):
        max_inter_score = 0
        max_inter_set = self.hier_tree[0][0][0]

        for level in range(1, self.max_level+1):
            for fea_set, score in self.hier_tree[level]:
                timeshap_set = [-(f-self.fea_num) for f in fea_set]
                for fea in timeshap_set:
                    score -= self.m[f'm{fea}']
                if abs(score) > max_inter_score:
                    max_inter_score = abs(score)
                    max_inter_set = fea_set
        
        return max_inter_set
    
    def shapley_topdown_tree(self):
        fea_num = self.fea_num
        fea_set = [list(range(fea_num))]
       
        self.hier_tree = {}
        self.hier_tree[0] = [(fea_set[0], self.v(fea_set[0]))]

        for level in range(1, self.fea_num):
            pos = 0
            min_inter_score = 1e8
            while pos < len(fea_set):
                subset = fea_set[pos]
                sen_len = len(subset)
                if sen_len == 1:
                    pos += 1
                    continue
                new_fea_set = [ele for x, ele in enumerate(fea_set) if x != pos]
                score_buff = []
                for idx in range(1, sen_len):
                    leave_one_set = deepcopy(new_fea_set)
                    sub_set1 = subset[0:idx]
                    sub_set2 = subset[idx:]
                    leave_one_set.insert(pos, sub_set1)
                    leave_one_set.insert(pos + 1, sub_set2)
                    score_buff.append(self.mobius_interaction(leave_one_set, pos, pos + 1))
                inter_score = np.array(score_buff)
                min_inter_idx = np.argmin(inter_score)
                minter = inter_score[min_inter_idx]
                if minter < min_inter_score:
                    min_inter_score = minter
                    inter_idx_opt = min_inter_idx
                    pos_opt = pos
                pos += 1

            new_fea_set = [ele for x, ele in enumerate(fea_set) if x != pos_opt]
            subset = fea_set[pos_opt]
            sub_set1 = subset[0:inter_idx_opt + 1]
            sub_set2 = subset[inter_idx_opt + 1:]
            new_fea_set.insert(pos_opt, sub_set1)
            new_fea_set.insert(pos_opt + 1, sub_set2)
            fea_set = new_fea_set
            new_level = []
            for fea in fea_set:
                new_level.append((fea, self.v(fea)))
            self.hier_tree[level] = new_level
        self.max_level = level
        return self.hier_tree
