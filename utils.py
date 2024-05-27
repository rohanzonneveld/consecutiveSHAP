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

class MyCompleteMobiusKernel(TimeShapKernel):
    def __init__(self, model, background, rs, mode, output_path=None):
        super().__init__(model, background, rs, mode)
        self.output_path = output_path
    
    def solve(self, fraction_evaluated, dim):

        if self.output_path: self.output_path += '/true'
        ## calculate true mobius transforms analytically
        # mobius transform ---------------------------------------------------------------------------------------------------------------------------
        N = np.vstack((np.ones(self.M), self.maskMatrix)) # full set, all coalitions
        v = np.hstack((self.fx[dim] - self.fnull[dim], self.ey[:, dim] - self.fnull[dim])) # create corresponding values for characteristic function

        m = np.zeros(N.shape[0])
        terms = []
        # iterate over all coalitions in N
        for i in tqdm.tqdm(range(N.shape[0])):
            # find features in coalition
            inds = np.nonzero(N[i, :])[0]
            # create term for coalition
            terms += [f'm{".".join([str(int(idx)+1) for idx in inds])}']
            # find size of coalition S
            S = len(inds)
            # iterate over all subsets of coalition S including S but excluding the empty set
            for T in range(1, S+1):
                # create all subsets of size T
                for sub_inds in itertools.combinations(inds, T):
                    # create subset
                    subset = np.zeros(N.shape[1])
                    subset[list(sub_inds)] = 1
                    # find index of subset in N
                    j = np.where((N==subset).all(axis=1))[0][0]
                    # calculate mobius transform from characteristic function v
                    m[i] += (-1)**(S-T) * v[j]
        
        # calculate shapley value from mobius transform -------------------------------------------------------------------------------------------
        # initialize shapley value vector
        phi = np.zeros(self.M)

        # go over each mobius term and distribute value equally over all events in term
        m_list = []
        for term, mobius in zip(terms, m):
            events = term.split('.')
            events[0] = events[0][1:]
            for event in events:
                phi[int(event)-1] += mobius/len(events)

            m_list += [[term, mobius]]

        # save for evaluation ---------------------------------------------------------------------------------------------------------------------
        if self.output_path:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

            ## Mobius Transforms
            pd.DataFrame(m_list, columns=['Term', 'Mobius Transform']).sort_values(by='Term').to_csv(f"{self.output_path}/mobius_transforms.csv", index=False)

            ## Shapley Values
            pd.DataFrame(phi, columns=['Shapley Value']).to_csv(f"{self.output_path}/shapley_values.csv", index=False)
 
        return phi, np.ones(len(phi))

    def _find_all_interactions(self, X):

        
        interactions = np.zeros((X.shape[0], self.n_terms), dtype='float32')
        interactions[:, :self.M] = X
        idx = self.M

        terms = [f'm{i}' for i in range(1, self.M+1)]
        next_terms = terms[:-1].copy()

        for order in range(2, self.M+1):
            new_terms = []
            for term in next_terms:
                row = []
                for m in range(1, self.M+1):
                    if m>int(term[1:].split('.')[-1]):
                        terms.append(term + f'.{m}')
                        row.append(term + f'.{m}')
                        inds = terms[-1].split('.')
                        inds[0] = inds[0][1:]
                        interactions[:, idx] = np.prod(X[:, [int(i)-1 for i in inds]], axis=1)
                        idx += 1
                if row: new_terms += row[:-1]
            next_terms = new_terms

        return interactions, terms


class FastConsecutiveMobiusKernel(TimeShapKernel):
    def __init__(self, model, background, rs, mode, output_path=None, lam=0.1):
        super().__init__(model, background, rs, mode)
        self.lam = lam

    def solve(self, fraction_evaluated, dim):

        eyAdj = self.linkfv(self.ey[:, dim]) - self.link.f(self.fnull[dim])
        mat = self.maskMatrix.copy()
        mat = np.append(mat, np.array([np.ones((self.M)), np.zeros((self.M))]), axis=0)
        consecutive_inner_prod = self.consecutive_inner_prod()

        sample_weights = np.append(np.ones((self.nsamples,)), [100000000]*2, axis=0)
        Omega = 2**consecutive_inner_prod - 1
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

    def consecutive_inner_prod(self, X):
        # TODO: create a function that calculates the inner product of a matrix with its transpose only with consecutive elements

        # example:
        # a   =    [a11, a12
        #  	        a21, a22, 
        #  	        a31, a32]

        # a.T =    [a11, a21, a31,
        #       	a12, a22, a32]

        # a*a.T=   [a11*a11 + a12*a12,	a11*a21 + a12*a22,	a11*a31 + a12*a32,
        #
        # 	        a21*a11 + a22*a12,	a21*a21 + a22*a22,	a21*a31 + a22*a32,
        #   
        #  	        a31*a11 + a32*a12,	a31*a21 + a32*a22, 	a31*a31 + a32*a32]


        #      =   [a11*a11 + a12*a12,	a11*a21 + a12*a22,	          a12*a32,
        #
        # 	        a21*a11 + a22*a12,	a21*a21 + a22*a22,	          a22*a32,
        #   
        #  	                  a32*a12,	          a32*a22, 	          a32*a32]

        inner_prod = X @ X.T
        return inner_prod


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
    def __init__(self, model, data, baseline=None, pruning_idx=0, win_size=5):
        score = model(data)[0][0]
        self.pred_label = round(score)
   
        self.pruning_idx = pruning_idx
        self.fea_num = data.shape[1] - self.pruning_idx

        self.model = model
        self.data = data
        self.win_size = win_size #TODO: not needed with mobius

        if baseline is None:
            self.baseline = np.expand_dims(np.zeros(data.shape[1]), axis=[0,2])
        else:
            self.baseline = baseline
        self.bias = model(baseline)[0][0]
        self.valdict = dict()


    def set_contribution_func(self, fea_set):
        # input has just one sentence, input is a list
        flat_fea = [f for fea in fea_set for f in fea]
        flat_fea = frozenset(flat_fea)
        if flat_fea in self.valdict:
            return self.valdict[frozenset(flat_fea)]
        
        data = self.data
        mask_data = np.zeros(data.shape[1])
        
        for fea_idx in fea_set:
                for idx in fea_idx:
                    mask_data[self.pruning_idx + idx] = data[0, self.pruning_idx+ idx, 0]

        mask_data = np.expand_dims(mask_data, axis=[0,2])
        score = self.model(mask_data)[0][0]
        contri = score - self.bias
        self.valdict[frozenset(flat_fea)] = contri
        
        return contri

    def get_shapley_interaction_weight(self, d, s):
        return np.math.factorial(s) * np.math.factorial(d - s - 2) / np.math.factorial(d - 1) / 2

    def shapley_interaction_approx(self, feature_set, left, right):
        win_size = self.win_size
        if left + 1 != right:
            print("Not adjacent interaction")
            return -1
        fea_num = len(feature_set)
        curr_set_lr = [feature_set[left], feature_set[right]]
        curr_set_l = [feature_set[left]]
        curr_set_r = [feature_set[right]]
        if left + 1 - win_size > 0:
            left_set = feature_set[left - win_size:left]
        else:
            left_set = feature_set[0:left]
        if right + win_size > fea_num - 1:
            right_set = feature_set[right + 1:]
        else:
            right_set = feature_set[right + 1:right + win_size + 1]
        adj_set = left_set + right_set
        num_adj = len(adj_set)
        dict_subset = {r: list(combinations(adj_set, r)) for r in range(num_adj + 1)}
        
        for i in range(num_adj + 1):
            weight = self.get_shapley_interaction_weight(fea_num, i)
            if i == 0:
                score_included = self.set_contribution_func(curr_set_lr)
                score_excluded_l = self.set_contribution_func(curr_set_r)
                score_excluded_r = self.set_contribution_func(curr_set_l)
                score_excluded = self.set_contribution_func([[]])
                score = (score_included - score_excluded_l - score_excluded_r + score_excluded) * weight
            else:
                for subsets in dict_subset[i]:
                    score_included = self.set_contribution_func(list(subsets) + curr_set_lr)
                    score_excluded_l = self.set_contribution_func(list(subsets) + curr_set_r)
                    score_excluded_r = self.set_contribution_func(list(subsets) + curr_set_l)
                    score_excluded = self.set_contribution_func(list(subsets))
                    score += (score_included - score_excluded_l - score_excluded_r + score_excluded) * weight
        return score

    def shapley_topdown_tree(self):
        fea_num = self.fea_num
        if fea_num == 0:
            return -1
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
                    score_buff.append(self.shapley_interaction_approx(leave_one_set, pos, pos + 1))
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

    def get_importance_phrase(self):
        phrase_dict = dict()
        for level in range(1,self.max_level+1):
            for fea_set, score in self.hier_tree[level]:
                phrase_dict[frozenset(fea_set)] = score
        phrase_tuple = sorted(phrase_dict.items(), key=lambda item: item[1], reverse=True)
        phrase_list = [list(item[0]) for item in phrase_tuple]
        score_list = [item[1] for item in phrase_tuple]
        return phrase_list, score_list


    def visualize_tree(self, wordvocab, fontsize=8, folder='model/now', tag='x'):
        levels = self.max_level

        cnorm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)
        cmapper = mpl.cm.ScalarMappable(norm=cnorm, cmap='RdYlBu_r')

        words = self.data[0, self.pruning_idx:, 0]
        fig, ax = plt.subplots(figsize=(12, 7))
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
        value = self.bias+self.m0
        fea_color = cmapper.to_rgba(value)
        ax.barh(height, width=width, height=0.5, left=start, color=fea_color)
        ax.text(start + width / 2, height, str(round(value, 2)), ha='center', va='center', color=text_color, fontsize=12)
        cb = fig.colorbar(cmapper, ax=ax)
        cb.ax.tick_params(labelsize=18)

        path = f'experiments/HEDGE/{folder}'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}/visualization_sentence_{tag}.png')
        # plt.show()


class MobiusHEDGE(HEDGE): # note, events in mobius are counted in opposite direction from HEDGE (1 is last event in mobius, 0 is first event in HEDGE) 
    def __init__(self, model, data, baseline=None, pruning_idx=0, win_size=None):
        super().__init__(model, data, baseline=baseline, pruning_idx=pruning_idx, win_size=win_size)
        explainer = ConsecutiveMobiusKernel(model, baseline, 42)
        self.shap_values = explainer.shap_values(data, nsamples=2048, pruning_idx=pruning_idx) 
        self.m = explainer.mobius_transforms
        self.m0 = explainer.m0
        print('mobius transform calculated')

    def shapley_interaction_approx(self, feature_set, left, right):
        return self.mobius_interaction(feature_set, left, right)

        'Calculate Shapley Interaction Index with mobius transform approximation'	

        left = feature_set[left]
        right = feature_set[right]

        X = set([f for fea in feature_set for f in fea])
        T = set(left+right)
        n = len(X)

        sii = 0
        for size in range(n-len(T)):
            for K in itertools.combinations(X-T, size):
                TK = set(K).union(T)
                name = 'm' + '.'.join([str(i+1) for i in TK])
                m = self.m[name] if name in self.m else 0 # consecutive assumption
                sii += 1/(size + 1) * m
        return sii

    def v(self, features):
        'value wrt bias of a span of features calculated with mobius transform'

        # because events are counted in opposite direction from HEDGE, we need to reverse the order of features
        # example: HEDGE features = [0, 1, 2, 3, 4], Mobius features = [5, 4, 3, 2, 1]
        #          HEDGE features = [1, 2, 3], Mobius features = [4, 3, 2]
        features = sorted([-(f-self.fea_num) for f in features])

        # value = 0
        # for size in range(1, len(features)+1):
        #     for coal in itertools.combinations(features, size):
        #         name = 'm' + '.'.join([str(i) for i in coal])
        #         if name in self.m:
        #             value += self.m[name]

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

        # value = 0
        # for span in feature_set:
        #     value += self.v(span)

        # # TODO get previous value from hier_tree

        # # level_up = feature_set[:left] + [feature_set[left] + feature_set[right]] + feature_set[right + 1:]

        # # prior_value = 0
        # # for span in level_up:
        # #     prior_value += self.v(span)
        # prior_level = list(self.hier_tree.keys())[-1]
        # prior_value = self.hier_tree[prior_level][1]

        value = self.v(feature_set[left]) + self.v(feature_set[right])

        prior_level = list(self.hier_tree.keys())[-1]
        prior_value = self.hier_tree[prior_level][left][1]

        return abs(prior_value - value)

    def compute_shapley_hier_tree(self):
        hier_tree = self.shapley_topdown_tree()
        self.hier_tree = {}
        for level in range(self.max_level+1):
            self.hier_tree[level] = []
            for idx, subset in enumerate(hier_tree[level]):
                score = self.v(subset)
                self.hier_tree[level].append((subset,score))
        return self.hier_tree
    
    def find_highest_interaction(self):
        max_inter_score = 0
        for level in range(1, self.max_level+1):
            for fea_set, score in self.hier_tree[level]:
                timeshap_set = [-(f-self.fea_num) for f in fea_set]
                for fea in timeshap_set:
                    score -= self.m[f'm{fea}']
                if abs(score) > max_inter_score:
                    max_inter_score = abs(score)
                    max_inter_set = fea_set
        
        return max_inter_set

        

