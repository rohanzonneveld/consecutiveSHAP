from timeshap.explainer.kernel import TimeShapKernel
from typing import Callable, List, Union
import numpy as np
import pandas as pd
import copy
import logging
import itertools
import scipy as sp

from timeshap.utils.timeshap_legacy import time_shap_match_instance_to_data
from shap.utils._legacy import convert_to_instance
from scipy.special import binom

import warnings

import sklearn
from packaging import version
from sklearn.linear_model import Lasso, LassoLarsIC, lars_path, Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger('shap')



log = logging.getLogger('shap')



class ConsecutiveKernel(TimeShapKernel):
    def __init__(self, model, background, rs, mode, maxOrder=1):
        super().__init__(model, background, rs, mode)
        self.maxOrder = maxOrder
    
    def shap_values(self, X, pruning_idx=None, **kwargs):
        assert isinstance(X, np.ndarray), "Instance must be 3D numpy array"
        if self.mode == "pruning":
            assert pruning_idx is not None
        else:
            assert pruning_idx < X.shape[1], "Pruning idx must be smaller than the sequence length. If not all events are pruned"
        assert pruning_idx % 1 == 0, "Pruning idx must be integer"
        self.pruning_idx = int(pruning_idx)

        self.set_variables_up(X)

        if sp.sparse.issparse(X) and not sp.sparse.isspmatrix_lil(X):
            X = X.tolil()

        # single instance
        if X.shape[0] == 1:
            # calculate mobius transform for all (consecutive) terms
            m = self.mobius_transform(X, **kwargs) 
            sizes=[0]
            for i in range(1,self.maxOrder+1):
                sizes = np.hstack((sizes, np.ones(self.M) * i))

            # calculate shapley value from mobius transform
            phi = np.zeros(self.M)
            for i in range(self.M): # TODO: pruned events?
                # find all mobius transforms that contain feature i, weighted sum is shapley value
                S = np.zeros(self.M)
                
                # TODO: find all elements of m where i is present
                # we want to find all sets where i is present in the coalition, so we create a mask with features present
                # begin = 0 if i < self.maxOrder else i - self.maxOrder
                # end = self.M-1 if i > self.M - self.maxOrder else i + self.maxOrder
                # S[begin:end] = 1

                inds = np.where(self._find_consecutive_features(S)==1)[0]
                phi[i] = np.sum(m[inds+1]/sizes[inds+1]) # m(S) consists of all m terms that correspond to the coalition S, +1 because first element is m0

                
            return phi

    def mobius_transform(self, X, **kwargs):
 
        explanation = self.explain(X, **kwargs)

        # display_events = ["m0"]
        # display_events += [f"m{str(int(i))}" for i in np.arange(1, X.shape[1]-self.pruning_idx+1)]

        # if self.pruning_idx > 0:
        #     display_events += ["Pruned Events"]

        # for order in range(1, self.maxOrder): 
        #     for i in np.arange(1, X.shape[1]-self.pruning_idx+1-order):
        #         name = ['m']
        #         name += '.'.join([str(int(j)) for j in np.arange(i, i+order+1)])
        #         display_events += [''.join(name)]

        # m = {}
        # for mobius, term in zip(explanation, display_events):
        #     m[term] = mobius[0] 
        m = explanation

        return m

    
    def explain(self, incoming_instance, **kwargs):
        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        instance.group_display_values = self.data.group_names
        time_shap_match_instance_to_data(instance, self.data)


        if self.mode == "feature":
            if self.pruning_idx > 0:
                self.varyingInds = self.varying_groups(instance.x, self.data.groups_size - 1) # R: exclude pruned events
                # add an index for pruned events
                self.varyingInds = np.concatenate((self.varyingInds, np.array([self.data.groups_size - 1])))
            else:
                self.varyingInds = self.varying_groups(instance.x, self.data.groups_size)
        
        elif self.mode == "event":
            self.varyingInds = np.array([x for x in np.arange(incoming_instance.shape[1]-1, self.pruning_idx-1, -1)]) # R: start from the most recent in reversed order until pruned is reached

        else:
            raise ValueError("`explain` -> mode not suported")

        if self.data.groups is None:
            self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
            self.M = self.varyingFeatureGroups.shape[0]
        else:
  
            if self.mode in ['feature']:
                if self.pruning_idx > 0:
                    self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds[:-1]]
                else:
                    self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
                self.M = len(self.varyingFeatureGroups)
                if self.pruning_idx > 0:
                    self.M += 1

            elif self.mode in ['event']:
                self.varyingFeatureGroups = self.varyingInds
                self.M = len(self.varyingFeatureGroups)
                if self.pruning_idx > 0:
                    self.M += 1


            else:
                raise ValueError("`explain` -> mode not suported")
            
            groups = self.data.groups
            # convert to numpy array as it is much faster if not jagged array (all groups of same length)
            if isinstance(self.varyingFeatureGroups, list) and all(len(groups[i]) == len(groups[0]) for i in range(len(self.varyingFeatureGroups))):
                self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
                # further performance optimization in case each group has a single value
                if self.varyingFeatureGroups.shape[1] == 1:
                    self.varyingFeatureGroups = self.varyingFeatureGroups.flatten()

        if self.returns_hs:
            # Removed the input variability to receive pd.series and DataFrame
            model_out, _ = self.model.f(instance.x)
        else:
            model_out = self.model.f(instance.x)

        self.fx = model_out[0]
        if not self.vector_out:
            self.fx = np.array([self.fx])

        #explained_score = (self.fx - self.fnull)[0]
        # if abs(explained_score) < 0.1:
        #     raise ValueError(f"Score difference between baseline and instance ({explained_score}) is too low < 0.1."
        #                      f"Baseline score: {self.fnull[0]} | Instance score: {self.fx[0]}."
        #                      f"Consider choosing another baseline.")

        # if no features vary then no feature has an effect
        if self.M == 0:
            phi = np.zeros((self.data.groups_size, self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((self.data.groups_size, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull) # R: v(fx) - v(fnull)
            for d in range(self.D):
                phi[self.varyingInds[0], d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:
            self.l1_reg = kwargs.get("l1_reg", "auto") # R: checks if l1_reg is passed else auto

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2**11

            # if we have enough samples to enumerate all subsets then ignore the unneeded samples
            self.max_samples = 2 ** 30
            if self.M <= 30:
                self.max_samples = 2 ** self.M - 2 # R: -2 to delete full and empty set
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            self.allocate()

            # weight the different subset sizes, R: this is the kernel, weights are used in addSample, TODO: add interactions here? (kernelize interactions)
            num_subset_sizes = int(np.ceil((self.M - 1) / 2.0)) # R: how many subset sizes to iterate over, only half because complement is added in the loop
            num_paired_subset_sizes = int(np.floor((self.M - 1) / 2.0)) # R: how many subset sizes have a complement (e.g if M=5 1-4 and 2-3 are pairs)
            weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)]) # R: shapley kernel, M choose |z|(i) done in line 201
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)
            log.debug("weight_vector = {0}".format(weight_vector))
            log.debug("num_subset_sizes = {0}".format(num_subset_sizes))
            log.debug("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
            log.debug("M = {0}".format(self.M))

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype='int64')
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):

                # determine how many subsets (and their complements) are of the current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes: nsubsets *= 2 # R: multiply by 2 to account for complements
                log.debug("subset_size = {0}".format(subset_size))
                log.debug("nsubsets = {0}".format(nsubsets))
                log.debug("self.nsamples*weight_vector[subset_size-1] = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1]))
                log.debug("self.nsamples*weight_vector[subset_size-1]/nsubsets = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets))

                # see if we have enough samples to enumerate all subsets of this size
                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size) # R: weighting of current subset size, M choose |z|
                    if subset_size <= num_paired_subset_sizes: w /= 2.0 # R: convert back to weight of size if weighted for both size and complement
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype='int64')] = 1.0
                        self.add_sample(instance.x, mask, w)
                        if subset_size <= num_paired_subset_sizes: # R: add the complement sample, because of this all the lines about *2 and /2 make sense
                            mask[:] = np.abs(mask - 1)
                            self.add_sample(instance.x, mask, w)
                else:
                    break
            log.info("num_full_subsets = {0}".format(num_full_subsets))
            # add random samples from what is left of the subset space
            nfixed_samples = self.nsamplesAdded
            samples_left = self.nsamples - self.nsamplesAdded
            log.debug("samples_left = {0}".format(samples_left))
            np.random.seed(self.random_seed)
            if num_full_subsets != num_subset_sizes: # R: if we have not yet enumerated all subsets fill the rest with random samples
                remaining_weight_vector = copy.copy(weight_vector)
                remaining_weight_vector[:num_paired_subset_sizes] /= 2 # because we draw two samples each below
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                log.info("remaining_weight_vector = {0}".format(remaining_weight_vector))
                log.info("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
                ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
                ind_set_pos = 0
                used_masks = {}
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    ind = ind_set[ind_set_pos] # we call np.random.choice once to save time and then just read it here
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.add_sample(instance.x, mask, 1.0)
                    else:
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0

                    # add the compliment sample
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before, otherwise just
                        # increment a previous sample's weight
                        if new_sample:
                            samples_left -= 1
                            self.add_sample(instance.x, mask, 1.0)
                        else:
                            # we know the compliment sample is the next one after the original sample, so + 1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

                # normalize the kernel weights for the random samples to equal the weight left after
                # the fixed enumerated samples have been already counted
                weight_left = np.sum(weight_vector[num_full_subsets:])
                log.info("weight_left = {0}".format(weight_left))
                self.kernelWeights[nfixed_samples:] *= weight_left / self.kernelWeights[nfixed_samples:].sum()

            # execute the model on the synthetic samples we have created
            self.run()

            self.n_terms = np.sum(np.arange(self.M-1, self.M-1-self.maxOrder, -1))
            self.n_terms += 1 if self.pruning_idx > 0 else 0
            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            m = np.zeros((self.n_terms+1, self.D))
            for d in range(self.D):
                vm, m0 = self.solve(self.nsamples / self.max_samples, d) # fraction evaluated, dim --> this is the kernel (ridge?) regression
                m[0, d] = m0
                m[1:,d] = vm

        if not self.vector_out:
            m = np.squeeze(m, axis=1)

        return m
    
    def solve(self, fraction_evaluated, dim): 
        
        X = np.zeros((self.maskMatrix.shape[0]+2, self.n_terms))

        # add empty and full set with infinite weight to make sure axioms are satisfied
        X[:-2, :] = self._find_consecutive_features(self.maskMatrix) # evaluation data
        X[-2, :] = 0 # empty set
        X[-1, :] = 1 # full set

        y = np.zeros(self.maskMatrix.shape[0]+2)
        y[:-2] = self.ey[:, dim]
        y[-2] = self.link.f(self.fnull[dim])
        y[-1] = self.link.f(self.fx[dim])
        
        W = np.zeros(self.maskMatrix.shape[0]+2)
        W[:-2] = 1 #self.kernelWeights # for mobius transform all samples are weighted equally
        W[-2:] = 1e6 # set very high weight for the full and empty set to ensure that axioms are satisfied

        
        # regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_seed) # works fine
        # regressor = Lasso(alpha=1, fit_intercept=True, random_state=self.random_seed) # Lasso makes all mabius transforms 0
        regressor = LinearRegression(fit_intercept=True) # works fine
        regressor.fit(X, y, sample_weight=W)
        m = regressor.coef_
        m0 = regressor.intercept_

        print(f"prediction full set: {regressor.predict(X[-1, :].reshape(1, -1))[0]}")
        print(f"actual full set: {y[-1]}")
        print(f"sum of mobius transform: {np.sum(m) + m0}")

        print(f"prediction empty set: {regressor.predict(X[-2, :].reshape(1, -1))[0]}")
        print(f"actual empty set: {y[-2]}")
        print(f"m0: {m0}")

        return m, m0
    
    def _find_consecutive_features(self, X):
        
        if X.ndim == 1:
          
            for order in range(1, self.maxOrder):
                consecutives = np.zeros(self.M - 1 - order) # -1 because the pruned events are not considered for consecutive interactions
                for j in range(self.M - 1 - order):
           
                    consecutives[j] = np.prod([X[j:j+order+1]])
                
                X = np.concatenate((X, consecutives))

        else:

            for order in range(1, self.maxOrder):
                consecutives = np.zeros((X.shape[0], self.M - 1 - order)) # -1 because the pruned events are not considered for consecutive interactions
                for j in range(self.M - 1 - order):

                    consecutives[:, j] = np.prod(X[:, j:j+order+1], axis=1)
                X = np.concatenate((X, consecutives), axis=1)


        return X
            



    
def feature_level(f: Callable,
                  data: np.ndarray,
                  baseline: np.ndarray,
                  pruned_idx: int,
                  random_seed: int,
                  nsamples: int,
                  model_feats: List[Union[int, str]] = None,
                  ) -> pd.DataFrame:
    """Method to calculate feature level explanations

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: np.array
        Sequence to explain.

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    pruned_idx: int
        Index to prune the sequence. All events up to this index are grouped

    random_seed: int
        Used random seed for the sampling process.

    nsamples: int
        The number of coalitions for TimeSHAP to sample.

    model_feats: List[Union[int, str]]
        The list of feature names.
        If none is provided, "Feature 1" format is used

    Returns
    -------
    pd.DataFrame
    """
    if pruned_idx == -1:
        pruned_idx = 0

    explainer = ConsecutiveKernel(f, baseline, random_seed, "feature")
    shap_values = explainer.shap_values(data, pruning_idx=pruned_idx, nsamples=nsamples)

    if model_feats is None:
        model_feats = ["Feature {}".format(i) for i in np.arange(data.shape[2])]

    model_feats = copy.deepcopy(model_feats)
    if pruned_idx > 0:
        model_feats += ["Pruned Events"]

    ret_data = []
    for exp, feature in zip(shap_values, model_feats):
        ret_data += [[random_seed, nsamples, feature, exp]]
    return pd.DataFrame(ret_data, columns=['Random seed', 'NSamples', 'Feature', 'Shapley Value'])

def event_level(f: Callable,
                data: np.array,
                baseline: Union[np.ndarray, pd.DataFrame],
                pruned_idx: int,
                random_seed: int,
                nsamples: int,
                display_events: List[str] = None,
                maxOrder: int = 1
                ) -> pd.DataFrame:
    """Method to calculate event level explanations

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: np.array
        Sequence to explain.

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    pruned_idx: int
        Index to prune the sequence. All events up to this index are grouped

    random_seed: int
        Used random seed for the sampling process.

    nsamples: int
        The number of coalitions for TimeSHAP to sample.

    display_events: List[str]
        In-order list of event names to be displayed

    Returns
    -------
    pd.DataFrame
    """
    explainer = ConsecutiveKernel(f, baseline, random_seed, "event", maxOrder=maxOrder)
    phi = explainer.shap_values(data, pruning_idx=pruned_idx, nsamples=nsamples)

    if display_events is None:
        display_events = [f"Event -{str(int(i))}" for i in np.arange(1, data.shape[1]-pruned_idx+1)]

        if pruned_idx > 0:
            display_events += ["Pruned Events"]

    ret_data = []
    for exp, event in zip(phi, display_events):
        ret_data += [[random_seed, nsamples, event, exp]]
   
    return pd.DataFrame(ret_data, columns=['rs', 'nsamples', 'event', 'shapley value']) # rs, nsamples, event, shapley value

def add_interactions(X, cols, maxOrder=1, index_column='timestamp', mode='inner'):
    '''
    Creates consecutive interactions within features up to maxOrder
    '''
    


    # initialize empty list to store new features, features interactions of the same order are in the same list
    interactions = [X]

    for order in range(maxOrder):
        # initialize empty list to store the new features of the current order
        interactionsNextOrder = pd.DataFrame()
        # iterate over all the features of the current order
        for col in cols:
            # TODO: add outer interaction mode
            interactionsNextOrder[col] = interactions[order][col] * X[col].shift(-(order + 1))

            # delete non-consecutive interactions
            is_consecutive = X[index_column].shift(-(order+1)) - X[index_column] == order+1
            inds = np.where(~is_consecutive)[0]
            interactionsNextOrder.loc[inds, col] = np.nan

        interactions.append(interactionsNextOrder)

    # rename features to include the interaction order
    for order in range(1, maxOrder+1):
        interactions[order].columns = [f"{col}_interactionOrder{order}" for col in cols]

    # concatenate all interaction features
    interactions = pd.concat(interactions[1:], axis=1)
    # add interaction features to the original dataset
    X = pd.concat([X, interactions], axis=1)
    # remove the first maxOrder rows, as they contain NaNs
    # X = X.iloc[maxOrder:]


    return X

