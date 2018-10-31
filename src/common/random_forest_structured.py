import numpy as np
import time
import cPickle
import pdb
from multiprocessing import Pool
from sklearn.decomposition import PCA
try:
    from scipy.weave import inline
except:
    from weave import inline

def example_forest_params():
    '''
    Returns a dictionary of some example forest params
    '''
    return {
        'num_tests': 500,
        'min_sample_cnt': 5,
        'max_depth': 25,
        'num_trees': 4,
        'bag_size': 0.5,
        'train_parallel': True,
        'njobs': 3,
        'num_dims_for_pca': 100,
        'sub_sample_exs_pca': True,
        'num_exs_for_pca': 2500,
        'oob_score': True,
        'oob_importance': False}



class Node:

    def __init__(self, node_id, exs_at_node, impurity, probability, medoid_id, tree_id):
        self.node_id = node_id
        depth = np.floor(np.log2(node_id+1))
        # print "In tree %d \t node %d \t depth %d" % (int(tree_id), int(node_id), int(depth))
        self.exs_at_node = exs_at_node
        self.impurity = impurity
        self.num_exs = float(exs_at_node.shape[0])
        self.is_leaf = True
        self.info_gain = 0.0
        self.tree_id = tree_id

        # just saving the probability of class 1 for now
        self.probability = probability
        self.medoid_id = medoid_id

    def update_node(self, test_ind1, test_thresh, info_gain):
        self.test_ind1 = test_ind1
        self.test_thresh = test_thresh
        self.info_gain = info_gain
        self.is_leaf = False

    def find_medoid_id(self, y_local):
        mu = y_local.mean(0)
        mu_dist = np.sqrt(((y_local - mu[np.newaxis, ...])**2).sum(1))
        return mu_dist.argmin()

    def create_child(self, test_res, impurity, prob, y_local, child_type):
        # test_res is binary and the same shape[0] as y_local
        assert test_res.shape[0] == y_local.shape[0]
        assert self.exs_at_node.shape[0] == y_local.shape[0]

        # save absolute location in dataset
        inds_local = np.where(test_res)[0]
        inds = self.exs_at_node[inds_local]

        # work out which values of y will be at the child node, then take the medoid
        med_id = inds[self.find_medoid_id(y_local.take(inds_local, 0))]

        if child_type == 'left':
            self.left_node = Node(2*self.node_id+1, inds, impurity, prob, med_id, self.tree_id)
        elif child_type == 'right':
            self.right_node = Node(2*self.node_id+2, inds, impurity, prob, med_id, self.tree_id)

    def get_leaf_nodes(self):
        # returns list of all leaf nodes below this node
        if self.is_leaf:
            return [self]
        else:
            return self.right_node.get_leaf_nodes() + \
                   self.left_node.get_leaf_nodes()

    def test(self, X):
        return X[self.test_ind1] < self.test_thresh

    def get_compact_node(self):
        # used for fast forest
        if not self.is_leaf:
            node_array = np.zeros(4)
            # dims 0 and 1 are reserved for indexing children
            node_array[2] = self.test_ind1
            node_array[3] = self.test_thresh
        else:
            node_array = np.zeros(2)
            node_array[0] = -1  # indicates that its a leaf
            node_array[1] = self.medoid_id  # the medoid id
        return node_array


class Tree:

    def __init__(self, tree_id, tree_params):
        self.tree_id = tree_id
        self.tree_params = tree_params
        self.num_nodes = 0
        self.label_dims = 0  # dimensionality of label space

    def build_tree(self, X, Y, node):
        if (node.node_id < ((2**self.tree_params['max_depth'])-1)) and (node.impurity > 0.0) \
                and (self.optimize_node(np.take(X, node.exs_at_node, 0), np.take(Y, node.exs_at_node, 0), node)):
                self.num_nodes += 2
                self.build_tree(X, Y, node.left_node)
                self.build_tree(X, Y, node.right_node)
        else:
            depth = np.floor(np.log2(node.node_id+1))
            # print "Leaf node: In tree %d \t depth %d \t %d examples" % \
                # (int(self.tree_id), int(depth), node.exs_at_node.shape[0])

    def discretize_labels(self, y):

        # perform PCA
        # note this randomly reduces amount of data in Y
        y_pca = self.pca(y)

        # discretize - here binary
        # using PCA based method - alternative is to use kmens
        y_bin = (y_pca[:, 0] > 0).astype('int')

        return y_pca, y_bin

    def pca(self, y):

        # select a random subset of Y dimensions (possibly gives robustness as well as speed)
        rand_dims = np.sort(np.random.choice(y.shape[1], np.minimum(self.tree_params['num_dims_for_pca'], y.shape[1]), replace=False))
        y_dim_subset = y.take(rand_dims, 1)

        pca = PCA(n_components=1, svd_solver='randomized') # compute for all components

        # optional: select a subset of exs (not so important if PCA is fast)
        if self.tree_params['sub_sample_exs_pca']:
            rand_exs = np.sort(np.random.choice(y.shape[0], np.minimum(self.tree_params['num_exs_for_pca'], y.shape[0]), replace=False))
            pca.fit(y_dim_subset.take(rand_exs, 0))
            return pca.transform(y_dim_subset)

        else:
            # perform PCA
            return pca.fit_transform(y_dim_subset)

    def train(self, X, Y, extracted_from):
        # no bagging
        #exs_at_node = np.arange(Y.shape[0])
        # bagging
        num_to_sample = int(float(Y.shape[0])*self.tree_params['bag_size'])

        if extracted_from is None:
            exs_at_node = np.random.choice(Y.shape[0], num_to_sample, replace=False)
        else:
            ids = np.unique(extracted_from)
            ids_for_this_tree = \
                np.random.choice(ids.shape[0], int(float(ids.shape[0])*self.tree_params['bag_size']), replace=False)

            # http://stackoverflow.com/a/15866830/279858
            exs_at_node = []
            for this_id in ids_for_this_tree:
                exs_at_node.append(np.where(extracted_from == this_id)[0])

            exs_at_node = np.hstack(exs_at_node)

            exs_at_node = np.unique(np.array(exs_at_node))

            if exs_at_node.shape[0] > num_to_sample:
                exs_at_node = np.random.choice(exs_at_node, num_to_sample, replace=False)

        exs_at_node.sort()
        self.bag_examples = exs_at_node

        # compute impurity
        #root_prob, root_impurity = self.calc_impurity(0, np.take(Y, exs_at_node), np.ones((exs_at_node.shape[0], 1), dtype='bool'),
        #                                    np.ones(1, dtype='float')*exs_at_node.shape[0])
        # cheating here by putting root impurity to 0.5 - should compute it
        root_prob = 0.5
        root_impurity = 0.5
        root_medoid_id = 0

        # create root
        self.root = Node(0, exs_at_node, root_impurity, root_prob, root_medoid_id, self.tree_id)
        self.num_nodes = 1
        self.label_dims = Y.shape[1]  # dimensionality of label space

        # build tree
        self.build_tree(X, Y, self.root)

        self.num_feature_dims = X.shape[1]

        # make compact version for fast testing
        self.compact_tree, _ = self.traverse_tree(self.root, np.zeros(0))

        if self.tree_params['oob_score']:

            # oob score is cooefficient of determintion R^2 of the prediction
            # oob score is in [0, 1], lower values are worse
            # Make predictions for examples not in the bag
            oob_exes = np.setdiff1d(np.arange(Y.shape[0]), exs_at_node)
            pred_idxs = self.test(X[oob_exes, :])

            # Compare the prediction to the GT (must be careful - as only indices returned)
            pred_Y = Y[pred_idxs.astype(np.int32), :]
            gt_Y = Y[oob_exes, :]

            u = ((pred_Y - gt_Y)**2).sum()
            v = ((gt_Y - gt_Y.mean(axis=0))**2).sum()
            self.oob_score = (1- u/v)

    def traverse_tree(self, node, compact_tree_in):
        node_loc = compact_tree_in.shape[0]
        compact_tree = np.hstack((compact_tree_in, node.get_compact_node()))

        # no this assumes that the index for the left and right child nodes are the first two
        if not node.is_leaf:
            compact_tree, compact_tree[node_loc] = self.traverse_tree(node.left_node, compact_tree)
            compact_tree, compact_tree[node_loc+1] = self.traverse_tree(node.right_node, compact_tree)

        return compact_tree, node_loc

    def calc_importance(self):
        ''' borrows from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx
        '''

        self.feature_importance = np.zeros(self.num_feature_dims)

        '''visit each node...'''
        stack = []
        node = self.root
        while stack or not node.is_leaf:

            if node.is_leaf:
                node = stack.pop()
            else:
                left = node.left_node
                right = node.right_node

                if not right.is_leaf and not left.is_leaf:
                    self.feature_importance[node.test_ind1] += \
                        (node.num_exs * node.impurity -
                        right.num_exs * right.impurity -
                        left.num_exs * left.impurity)

                if not right.is_leaf:
                    stack.append(right)
                node = left

        self.feature_importance /= self.feature_importance.sum()
        return self.feature_importance
            # oob_exes = np.setdiff1d(np.arange(Y.shape[0]), exs_at_node)
            # self.oob_importance = np.zeros((X.shape[1]))

            # # Take each feature dimension in turn
            # for feature_idx in range(X.shape[1]):

            #     # permute this column
            #     to_use = np.random.choice(oob_exes, 100)
            #     this_X = X.copy()[to_use, :]

            #     permutation = np.random.permutation(this_X.shape[0])
            #     this_X[:, feature_idx] = this_X[permutation, feature_idx]

            #     # Compare the prediction to the GT (must be careful - as only indices returned)
            #     pred_Y = Y[self.test(this_X).astype(np.int32), :]
            #     gt_Y = Y[to_use, :]

            #     # maybe here do ssd? Not normalised but could be ok...
            #     self.oob_importance[feature_idx] = \
            #         ((pred_Y - gt_Y)**2).sum(axis=1).mean()

            # print self.oob_importance

    def test_fast(self, X, max_depth):
        op = np.zeros((X.shape[0]))
        tree = self.compact_tree  # work around as I don't think I can pass self.compact_tree

        #in memory: for non leaf  node - 0 is lchild index, 1 is rchild, 2 is dim to test, 3 is threshold
        #in memory: for leaf node - 0 is leaf indicator -1, 1 is the medoid id
         # && depth < max_depth
        # print tree.shape
        code = """
        int ex_id, node_loc, depth;
        for (ex_id=0; ex_id<NX[0]; ex_id++) {
            node_loc = 0;
            //depth = 0
            while (tree[node_loc] != -1) {
                if (X2(ex_id, int(tree[node_loc+2]))  <  tree[node_loc+3]) {
                    node_loc = tree[node_loc+1];  // right node
                }
                else {
                    node_loc = tree[node_loc];  // left node
                }
                //depth++;
            }
            op[ex_id] = tree[node_loc + 1];  // medoid id
        }
        """
        inline(code, ['X', 'op', 'tree', 'max_depth'])
        return op

    def compact_leaf_nodes(self):
        leaf_locations = np.where(self.compact_tree==-1)[0]
        return self.compact_tree[leaf_locations+1]

    def test(self, X, max_depth=np.inf):
        op = np.zeros(X.shape[0])
        # check out apply() in tree.pyx in scikitlearn

        # single dim test
        for ex_id in range(X.shape[0]):
            node = self.root
            depth = 0
            while not (node.is_leaf or depth >= max_depth):
                if X[ex_id, node.test_ind1] < node.test_thresh:
                    node = node.right_node
                else:
                    node = node.left_node
                depth += 1
            # return medoid id
            op[ex_id] = node.medoid_id
        return op

    def leaf_nodes(self):
        '''returns list of all leaf nodes'''
        return self.root.get_leaf_nodes()

    def calc_impurity(self, node_id, y_bin, test_res, num_exs):
        # TODO currently num_exs is changed to deal with divide by zero, fix this
        # if don't want to divide by 0 so add a 1 to the numerator
        invalid_inds = np.where(num_exs == 0.0)[0]
        num_exs[invalid_inds] = 1

        # estimate probability
        # just binary classification
        node_test = test_res * (y_bin[:, np.newaxis] == 1)

        prob = np.zeros((2, test_res.shape[1]))
        prob[1, :] = node_test.sum(axis=0) / num_exs
        prob[0, :] = 1 - prob[1, :]
        prob[:, invalid_inds] = 0.5  # 1/num_classes

        # binary classification
        #impurity = -np.sum(prob*np.log2(prob))  # entropy
        impurity = 1-(prob*prob).sum(0)  # gini

        num_exs[invalid_inds] = 0.0
        return prob, impurity

    def node_split(self, x_local):
        # left node is false, right is true
        # single dim test
        test_inds_1 = np.sort(np.random.random_integers(0, x_local.shape[1]-1, self.tree_params['num_tests']))
        x_local_expand = x_local.take(test_inds_1, 1)
        x_min = x_local_expand.min(0)
        x_max = x_local_expand.max(0)
        test_thresh = (x_max - x_min)*np.random.random_sample(self.tree_params['num_tests']) + x_min
        #valid_var = (x_max != x_min)

        test_res = x_local_expand < test_thresh

        return test_res, test_inds_1, test_thresh

    def optimize_node(self, x_local, y_local, node):
        # TODO if num_tests is very large could loop over test_res in batches
        # TODO is the number of invalid splits is small it might be worth deleting the corresponding tests

        # perform split at node
        test_res, test_inds1, test_thresh = self.node_split(x_local)

        # discretize label space
        y_pca, y_bin = self.discretize_labels(y_local)

        # count examples left and right
        num_exs_l = (~test_res).sum(axis=0).astype('float')
        num_exs_r = x_local.shape[0] - num_exs_l  # i.e. num_exs_r = test_res.sum(axis=0).astype('float')
        valid_inds = (num_exs_l >= self.tree_params['min_sample_cnt']) & (num_exs_r >= self.tree_params['min_sample_cnt'])

        successful_split = False
        if valid_inds.sum() > 0:
            # child node impurity
            prob_l, impurity_l = self.calc_impurity(node.node_id, y_bin, ~test_res, num_exs_l)
            prob_r, impurity_r = self.calc_impurity(node.node_id, y_bin, test_res, num_exs_r)

             # information gain - want the minimum
            num_exs_l_norm = num_exs_l/node.num_exs
            num_exs_r_norm = num_exs_r/node.num_exs
            #info_gain = - node.impurity + (num_exs_r_norm*impurity_r) + (num_exs_l_norm*impurity_l)
            info_gain = (num_exs_r_norm*impurity_r) + (num_exs_l_norm*impurity_l)

            # make sure we con only select from valid splits
            info_gain[~valid_inds] = info_gain.max() + 10e-10  # plus small constant
            best_split = info_gain.argmin()

            # if the info gain is acceptable split the node
            # TODO is this the best way of checking info gain?
            #if info_gain[best_split] > self.tree_params.min_info_gain:
            # create new child nodes and update current node
            node.update_node(test_inds1[best_split], test_thresh[best_split], info_gain[best_split])
            node.create_child(~test_res[:, best_split], impurity_l[best_split], prob_l[1, best_split], y_local, 'left')
            node.create_child(test_res[:, best_split], impurity_r[best_split], prob_r[1, best_split], y_local, 'right')

            successful_split = True

        return successful_split


def train_forest_helper(parameters_tuple):
    '''
    Parallel training helper - used to train trees in parallel
    '''
    t_id, seed, params = parameters_tuple
    print 'tree', t_id, X.shape[0], Y.shape[0]
    np.random.seed(seed)
    tree = Tree(t_id, params)
    tree.train(X, Y, extracted_from)
    return tree


def _init(X_in, Y_in, extracted_from_in):
    '''
    Each pool process calls this initializer. Here we load the array(s) to be
    shared into that process's global namespace
    '''
    global X, Y, extracted_from
    X = X_in
    Y = Y_in
    extracted_from = extracted_from_in


class Forest:

    def __init__(self, params):
        self.params = params
        self.trees = []

    def make_lightweight(self):
        # delete the clunky version of each tree, keeping just the compact one
        for tree in self.trees:
            tree.root = None

    def save(self, filename):
        # make lightweight version for saving
        self.make_lightweight()

        with open(filename, 'wb') as fid:
            cPickle.dump(self, fid)

    def train(self, X_local, Y_local, extracted_from_local=None):
        '''
        extracted_from is an optional array which defines a class label
        for each training example. if provided, the bagging is done at
        the level of np.unique(extracted_from)
        '''
        if np.any(np.isnan(X_local)):
            raise Exception('nans should not be present in training X')

        if np.any(np.isnan(Y_local)):
            raise Exception('nans should not be present in training Y')

        if self.params['train_parallel']:
            # TODO Can I make this faster by sharing the data?
            #print 'Parallel training'
            # need to seed the random number generator for each process
            seeds = np.random.random_integers(0, np.iinfo(np.int32).max, self.params['num_trees'])

            # # these are the arguments which are different for each tree
            per_tree_args = ((t_id, seeds[t_id], self.params)
                for t_id in range(self.params['num_trees']))

            # data which is to be shared across all processes are passed as initargs
            pool = Pool(processes=self.params['njobs'], initializer=_init,
                initargs=(X_local, Y_local, extracted_from_local))

            self.trees.extend(pool.imap(train_forest_helper, per_tree_args))

            # these are very important to clear up the memory issues
            pool.close()
            pool.join()


        else:
            #print 'Standard training'
            for t_id in range(self.params['num_trees']):
                print 'tree', t_id
                tree = Tree(t_id, self.params)
                tree.train(X_local, Y_local, extracted_from_local)
                self.trees.append(tree)
        #print 'num trees ', len(self.trees)

    def test(self, X, max_depth=np.inf):
        if np.any(np.isnan(X)):
            raise Exception('nans should not be present in test X')

        # ensuring X is 2D
        X = np.atleast_2d(X)

        # return the medoid id at each leaf
        op = np.zeros((X.shape[0], len(self.trees)))
        for tt, tree in enumerate(self.trees):
            op[:, tt] = tree.test_fast(X, max_depth)
        return op

    def delete_trees(self):
        del self.trees[:]

    def calc_importance(self):
        imp = [tree.calc_importance() for tree in self.trees]
        return np.vstack(imp).mean(axis=0)
