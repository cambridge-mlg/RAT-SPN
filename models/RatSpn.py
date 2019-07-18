import numpy as np
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class RatSpn(object):
    """
    Class implementing RatSpns.

    *******
    A PRIMER ON LINEAR AND LOG DOMAIN: while we are discussing SPNs, we are actually doing all computation on the
    log-domain, for numerical stability. Therefore, often we say e.g. products in the documentation and variable names,
    but use a + operation, since everything is done in the log-domain. Please convert back-and-forth and try to not get
    confused :)
    *******

    The basic input for this class is a region graph (see class RegionGraph), provided in layered form as
    region_graph_layers (as obtained by RegionGraph.make_layers()), i.e. a list, containing lists of i) regions and
    ii) partitions (see RegionGraph.py for defintions of region and partition). This list alternates between regions
    and partitions, i.e. region_graph_layers[0] is a list of regions, region_graph_layers[1] is a list of partitions,
    region_graph_layers[2] is a list of regions, ... Moreover, the layers are assumed to topologically sorted, i.e.
    if k >= l, it is guaranteed that regions (partitions) in layer k cannot be children of partitions (regions) in
    layer l.

    Recall from the paper, that every region is assigned a set of distributions.

    For leaf regions, we can provide any kind of distributions we like. Currently, factorized Gaussians and factorized
    discrete (categorical) distributions are provided -- see _make_gauss_layer() and _make_discrete_layer(). These
    distributions are represented with a (batch_size, num_input_distributions) tensor of log-probabilities.

    For internal regions, we use a set of sum nodes (mixture distributions). They are represented with a
    (batch_size, num_sums) tensor of log-probabilities. Sums are implemented in the log-domain (using logsumexp).

    Further, recall any partition is associated with a set of factorized distributions, obtained by taking all
    cross-products of distributions which are associated with its child regions. E.g. a partitions which splits its
    parent region into 2 sub-regions, and both regions are equipped with num_sums sum nodes, the number of product
    nodes associated with this partition is num_sums**2. Products are represented with (batch_size, X) tensor of
    log-probabilities, where X depends on the number of nodes in the child regions.
    """

    def __init__(self,
                 region_graph_layers,
                 num_classes,
                 provided_inputs=None,
                 num_sums=20,
                 discrete_leaves=False,
                 num_states=2,
                 num_input_distributions=20,
                 gauss_min_var=1.0,
                 gauss_max_var=1.0,
                 gauss_isotropic=True,
                 normalized_sums=True,
                 dropout_rate_input=None,
                 dropout_rate_sums=None,
                 lambda_discriminative=1.0,
                 kappa_discriminative=1.0,
                 optimizer='adam',
                 provided_learning_rate=0.001,
                 learning_rate_decay=0.99,
                 ARGS=None):

        self.region_graph_layers = region_graph_layers
        self.root_region = region_graph_layers[-1][0]
        self.num_classes = num_classes
        self.num_dims = len(self.root_region)
        self.provided_inputs = provided_inputs

        if ARGS is None:
            self.discrete_leaves = discrete_leaves
            self.num_states = num_states
            self.num_input_distributions = num_input_distributions

            self.gauss_min_var = gauss_min_var
            self.gauss_max_var = gauss_max_var
            self.gauss_isotropic = gauss_isotropic

            self.normalized_sums = normalized_sums
            self.num_sums = num_sums

            self.dropout_rate_input = dropout_rate_input
            self.dropout_rate_sums = dropout_rate_sums

            self.lambda_discriminative = lambda_discriminative
            self.kappa_discriminative = kappa_discriminative

            self.provided_learning_rate = provided_learning_rate
            self.learning_rate_decay = learning_rate_decay
            self.optimizer = optimizer

        else:
            self.discrete_leaves = ARGS.discrete_leaves
            self.num_states = ARGS.num_states
            self.num_input_distributions = ARGS.num_input_distributions

            self.gauss_min_var = ARGS.gauss_min_var
            self.gauss_max_var = ARGS.gauss_max_var
            self.gauss_isotropic = ARGS.gauss_isotropic

            self.normalized_sums = ARGS.normalized_sums
            self.num_sums = ARGS.num_sums

            self.dropout_rate_input = ARGS.dropout_rate_input
            self.dropout_rate_sums = ARGS.dropout_rate_sums

            self.lambda_discriminative = ARGS.lambda_discriminative
            self.kappa_discriminative = ARGS.kappa_discriminative

            self.provided_learning_rate = ARGS.provided_learning_rate
            self.learning_rate_decay = ARGS.learning_rate_decay
            self.optimizer = ARGS.optimizer

        ### dictionaries mapping regions to assigned distributions
        # maps a region to its log-probability tensor
        self.region_distributions = dict()
        self.region_distributions_string = dict()
        # maps a region to a list of log-probability tensors, one for each child partition of the region
        self.region_products = dict()

        ### inputs, outputs, layers
        # inputs, i.e. observed values. Missingness of values cam be expressed with the dropout masks
        # This is a (batch_size, num_dims) tensor of floats.
        self.inputs = None
        # outputs, i.e. the log-probability tensor of the root region
        self.outputs = None
        # layers, following the structure of the underlying regions graph.
        # layers[0] contains input log-distribution tensors
        # layers[1] contains log-product tensors
        # layers[2] contains log-sum tensors
        # ...
        self.layers = []
        # we sometimes need to know the batch size, so we extract it below. This could probably done in a different way.
        self.dynamic_batch_size = None
        # assigns a string to each probability tensor
        self.types = {}

        ### dropout rate placeholders and generated dropoutmasks
        # dropout_input_placeholder are dropout_sums_placeholder are scalars, representing dropoutrates for inputs and
        # sums, respectively
        self.dropout_input_placeholder = None
        # binary droputout mask for input distributions
        self.dropout_mask_input = None
        self.dropout_sums_placeholder = None
        # binary droput masks for sum children. This is a list of lists of binary masks. The outer list runs over sum
        # layers.
        self.dropout_masks_in_layer = []

        ### dictionaries mapping distribution tensors to their params
        self.sum_params = {}
        self.gauss_mean_params = {}
        self.gauss_var_params = {}
        self.discrete_params = {}
        self.all_params = []

        ### label placeholder
        self.labels = None

        ### objective ops
        self.cross_entropy = None
        self.cross_entropy_mean = None
        self.log_likelihood = None
        self.log_likelihood_mean = None
        self.neg_norm_ll = None
        self.objective = None
        self.num_correct = None

        ### learning rate ops
        self.learning_rate = None
        self.decrease_lr_op = None

        ### optimizer
        self.optimizer_op = None
        self.train_op = None

        ### book-keeping for EM
        self.em_reset_accums = None
        self.em_update_accums = None
        self.em_update_params = None

        self.EM_deriv_input_pl = None

        self.log_derivs = dict()
        self.local_inputs = {}
        self.local_parent_inputs = {}
        self.product_shapes = {}
        self.parent_products = {}
        self.sibling_dicts = {}
        self.product_idx_in_region = {}
        self.reshaped_prod_derivs = {}
        self.prod_names = {}
        self.weighted_children = {}

        self.expanded_region_distributions_1 = dict()
        self.expanded_region_distributions_2 = dict()

        # make spn
        self._make_spn()

    def __str__(self):
        ret = ""
        for l in self.layers:
            type_strings = list(set([self.types[t] for t in l]))
            if len(type_strings) != 1:
                raise AssertionError("multiple types in layer")
            ret += "Layer: {}\n".format(type_strings[0])
            ret += "   Num Tensors: {}\n".format(len(l))
            ret += "   Num Nodes:   {}\n".format(sum([int(t.shape[1]) for t in l]))

        return ret

    @staticmethod
    def _variable(name, shape, stddev):
        """Get a TF variable."""

        var = tf.get_variable(
            name,
            shape,
            initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),
            dtype=tf.float32)

        return var

    def _make_discrete_layer(self, regions):
        """
        Make distribution layer for discrete data.
        The data should assume integers, starting with 0.
        """

        layer_distributions = []

        with tf.variable_scope('discrete') as scope:
            for region_counter, region in enumerate(regions):
                local_size = len(region)
                region_list = sorted(list(region))

                with tf.variable_scope(str(region_counter)) as scope:

                    # cut out the observations for the scope of the current region
                    # cast to int32, since we assume discrete data
                    local_inputs = tf.cast(tf.gather(self.inputs, region_list, axis=1, name='local_vars'), tf.int32)

                    # parameters beta of discrete distribution (softmax)
                    betas = self._variable(
                        'betas',
                        shape=[local_size, self.num_states, self.num_input_distributions],
                        stddev=1e-1)
                    discrete = tf.nn.log_softmax(betas, 1, 'log_weights')

                    # To get the log-probability, we need to read out the value indicated by the data.
                    #
                    # To illustrate the following: tiled_range for local_size == 5 is
                    #
                    # [0, 1, 2, 3, 4,
                    #  0, 1, 2, 3, 4,
                    #  ...
                    #  0, 1, 2, 3, 4]
                    #
                    # where the number of rows is the batch size
                    #
                    tile_shape = tf.concat([self.dynamic_batch_size, [1]], 0)
                    tiled_range = tf.tile(tf.expand_dims(tf.range(local_size), 0), tile_shape)

                    # idx_into_discrete is a (batch_size x local_size) matrix of 2D indices,
                    # where the index at (i,j) is "j, value of X_j for i'th sample"
                    # we use that, to "index into" our discrete distributions with gather_nd
                    idx_into_discrete = tf.transpose(tf.stack([tiled_range, local_inputs]), [1, 2, 0])
                    # indicator_log_pmf_single is a (batch_size x local_size x num_discrete) tensor;
                    # its (i,j,k)'th entry is the log-probability of the k'th discrete distribution of the
                    # j'th variable for the i'th sample
                    indicator_log_pmf_single = tf.gather_nd(discrete, idx_into_discrete)

                    if self.dropout_mask_input is not None:
                        local_dropout = tf.gather(self.dropout_mask_input, region_list, axis=1, name='local_dropout')
                        local_dropout = tf.expand_dims(local_dropout, -1)
                        indicator_log_pmf_single = indicator_log_pmf_single * local_dropout

                    # take a product (sum in log-domain) over scope dimensions -> local independence assumption
                    indicator_log_pmf = tf.reduce_sum(indicator_log_pmf_single, 1, name='log_pmf')
                    distributions = indicator_log_pmf

                    self.local_inputs[region] = local_inputs
                    self.discrete_params[distributions] = betas
                    self.all_params.append(betas)
                    self.region_distributions[region] = distributions
                    self.region_distributions_string[region] = "discrete" + str(region_counter)
                    self.types[distributions] = "discrete"
                    layer_distributions.append(distributions)

        self.layers.append(layer_distributions)

    def _make_gauss_layer(self, regions):
        """Make Gaussian layer."""

        layer_distributions = []

        with tf.variable_scope('gaussians') as scope:
            for region_counter, region in enumerate(regions):
                local_size = len(region)
                region_list = sorted(list(region))

                with tf.variable_scope(str(region_counter)) as scope:

                    # cut out the observations for the scope of the current region
                    local_inputs = tf.gather(self.inputs, region_list, axis=1, name='local_vars')

                    means = self._variable('means',
                                           shape=[1, local_size, self.num_input_distributions],
                                           stddev=1e-1)

                    var_params = None
                    # if gauss_min_var >= gauss_max_var, we just assume the identity matrix as covariance
                    if self.gauss_min_var < self.gauss_max_var:
                        if self.gauss_isotropic:
                            var_param_shape = [1, 1, self.num_input_distributions]
                        else:
                            var_param_shape = [1, local_size, self.num_input_distributions]

                        var_params = self._variable_with_weight_decay('sigma_params',
                                                                      shape=var_param_shape,
                                                                      stddev=1e-1)
                        min_log_var = np.log(self.gauss_min_var)
                        max_log_var = np.log(self.gauss_max_var)
                        log_var = min_log_var + (max_log_var - min_log_var) * tf.sigmoid(var_params)

                        local_x = tf.expand_dims(local_inputs, -1)
                        l2pi = np.log(2 * np.pi)
                        gauss_log_pdf_single = -0.5 * ((local_x - means)**2 / tf.exp(log_var) + l2pi + log_var)
                    else:
                        local_x = tf.expand_dims(local_inputs, -1)
                        l2pi = np.log(2 * np.pi)
                        gauss_log_pdf_single = -0.5 * ((local_x - means) ** 2 + l2pi)

                    if self.dropout_mask_input is not None:
                        local_dropout = tf.gather(self.dropout_mask_input, region_list, axis=1, name='local_dropout')
                        local_dropout = tf.expand_dims(local_dropout, -1)
                        gauss_log_pdf_single = gauss_log_pdf_single * local_dropout

                    gauss_log_pdf = tf.reduce_sum(gauss_log_pdf_single, 1, name='log_pdf')
                    distributions = gauss_log_pdf

                    self.local_inputs[region] = local_inputs
                    self.gauss_mean_params[gauss_log_pdf] = means
                    self.all_params.append(means)
                    if var_params:
                        self.gauss_var_params[gauss_log_pdf] = var_params
                        self.all_params.append(var_params)
                    self.region_distributions[region] = distributions
                    self.region_distributions_string[region] = "gaussian" + str(region_counter)
                    self.types[distributions] = "Gaussian"
                    layer_distributions.append(distributions)

        self.layers.append(layer_distributions)

    def _make_prod_layer(self, ps_count, partitions):
        """Make a product layer, by taking cross-products of all distributions, for all partitions."""

        layer_distributions = []

        with tf.variable_scope('products{}'.format(ps_count)) as scope:
            for partition in partitions:
                sub_region1 = partition[0]
                sub_region2 = partition[1]

                if sub_region1 > sub_region2:
                    sub_region1, sub_region2 = sub_region2, sub_region1

                # get the distributions associated with our sub-regions
                sub_dist1 = self.region_distributions[sub_region1]
                sub_dist2 = self.region_distributions[sub_region2]

                num_dist1 = int(sub_dist1.shape[1])
                num_dist2 = int(sub_dist2.shape[1])

                string1 = self.region_distributions_string[sub_region1]
                string2 = self.region_distributions_string[sub_region2]

                # we take all cross-products, thus expand in different dims and use broadcasting
                if sub_dist1 not in self.expanded_region_distributions_1:
                    sub_dist1_expand = tf.expand_dims(sub_dist1, 1, name=string1 + "_expand1")
                    self.expanded_region_distributions_1[sub_dist1] = sub_dist1_expand
                else:
                    sub_dist1_expand = self.expanded_region_distributions_1[sub_dist1]

                if sub_dist2 not in self.expanded_region_distributions_2:
                    sub_dist2_expand = tf.expand_dims(sub_dist2, 2, name=string2 + "_expand2")
                    self.expanded_region_distributions_2[sub_dist2] = sub_dist2_expand
                else:
                    sub_dist2_expand = self.expanded_region_distributions_2[sub_dist2]

                prod_name = string1 + "X" + string2
                with tf.variable_scope(prod_name) as scope:

                    # product == sum in log-domain
                    prod = sub_dist1_expand + sub_dist2_expand

                    # remember the original shape of product, as we require this for the EM backwards pass
                    orig_shape = tf.shape(prod)
                    # flatten the outer product
                    prod = tf.reshape(prod, [-1, num_dist1 * num_dist2], name="prods")

                    # remember the product tensor for the super-region
                    super_region = tuple(sorted(sub_region1 + sub_region2))
                    region_products = self.region_products.get(super_region, [])
                    self.region_products[super_region] = region_products + [prod]

                    ### make book-keeping required for the EM backwards pass
                    # remember the product shape stemming from broadcasting, as we also broadcast in backwards-pass
                    self.product_shapes[prod] = orig_shape

                    # remember parent product tensors for sub-regions
                    parent_products = self.parent_products.get(sub_region1, [])
                    self.parent_products[sub_region1] = parent_products + [prod]
                    parent_products = self.parent_products.get(sub_region2, [])
                    self.parent_products[sub_region2] = parent_products + [prod]

                    # remember product siblings; we store them in the shape used for broadcasting
                    sibling_dict1 = self.sibling_dicts.get(sub_region1, {})
                    sibling_dict1[prod] = (sub_dist2_expand, 2)
                    self.sibling_dicts[sub_region1] = sibling_dict1

                    sibling_dict2 = self.sibling_dicts.get(sub_region2, {})
                    sibling_dict2[prod] = (sub_dist1_expand, 1)
                    self.sibling_dicts[sub_region2] = sibling_dict2

                    #
                    self.prod_names[prod] = prod_name

                    self.types[prod] = "product"
                    layer_distributions.append(prod)

        self.layers.append(layer_distributions)

    def _make_prod_backprop_layer(self, ps_count, regions):
        """"
        Make a layer computing the log-derivative of a prod layer, required for the EM backwards pass. EM in SPNs can
        be reduced to backprop, see Peharz et al., On the latent variable interpretation in sum-product networks,
        TPAMI 2017.

        The backprop signal at a product node is simply the sum of backprop signals from all its sum parents, times
        the corresponding sum-weights.

        This function insert the required computational ops to do this for all products, in all partitions, in all
        regions.
        """

        with tf.variable_scope('deriv_prods{}'.format(ps_count)) as scope:
            for region in regions:
                sums = self.region_distributions[region]
                log_weights = self.sum_params[sums]
                log_deriv_dist = self.log_derivs[sums]
                log_deriv_dist = tf.expand_dims(log_deriv_dist, 1)

                for prods in self.region_products[region]:
                    # product_idx_in_region[prods] addresses start-end in the weight in the current region
                    # they are set in _make_sum_layer()
                    cur_idx = self.product_idx_in_region[prods]
                    local_log_weights = tf.slice(log_weights, [0, cur_idx[0], 0], [-1, cur_idx[1]-cur_idx[0], -1])

                    self.log_derivs[prods] = tf.reduce_logsumexp(log_deriv_dist + local_log_weights, 2)

                    # also make a reshaped version of the log-derivative, for EM
                    reshaped_prod_deriv = tf.reshape(self.log_derivs[prods], self.product_shapes[prods])
                    self.reshaped_prod_derivs[prods] = reshaped_prod_deriv

    def _make_sum_layer(self, ps_count, regions, num_sums_per_region, dropout_op):
        """Make a sum layer, by taking mixtures of all products with the same scope."""

        layer_distributions = []
        dropout_masks = []

        with tf.variable_scope('sums{}'.format(ps_count)) as scope:
            for region_counter, region in enumerate(regions):
                if region not in self.region_products:
                    raise AssertionError('No products found for region.')

                with tf.variable_scope(str(region_counter)) as scope:

                    # bookkeeping for backprop/EM
                    # product_idx_in_region stores the start-end address within the sum-weight tensor, for each
                    # product tensor.
                    start_idx = 0
                    for sub_prods in self.region_products[region]:
                        num_prods = int(sub_prods.shape[1])
                        self.product_idx_in_region[sub_prods] = (start_idx, start_idx + num_prods)
                        start_idx += num_prods

                    prods = tf.concat(self.region_products[region], 1, name="collected_prods")
                    prods = tf.expand_dims(prods, axis=-1)

                    with tf.variable_scope('log_weights') as scope:
                        params = self._variable('params',
                                                shape=[1, prods.shape[1], num_sums_per_region],
                                                stddev=5e-1)

                        if self.normalized_sums:
                            weights = tf.nn.log_softmax(params, 1, 'log_weights')
                        else:
                            weights = params

                        if dropout_op is not None:
                            random_tensor = random_ops.random_uniform(tf.shape(prods), dtype=prods.dtype)
                            dropout_mask = tf.log(math_ops.floor(dropout_op + random_tensor))
                            prods = prods + dropout_mask

                            dropout_masks.append(dropout_mask)

                    weighted_children = tf.add(prods, weights, name="weighted_children")
                    sums = tf.reduce_logsumexp(weighted_children, axis=1, name="sums")

                    self.weighted_children[sums] = weighted_children
                    self.region_distributions[region] = sums
                    self.region_distributions_string[region] = "sums" + str(ps_count) + "_" + str(region_counter)
                    self.types[sums] = "sum"
                    self.sum_params[sums] = params
                    self.all_params.append(params)
                    layer_distributions.append(sums)

        self.layers.append(layer_distributions)
        self.dropout_masks_in_layer.append(dropout_masks)

    def _make_dist_backprop_layer(self, ps_count, regions, root_log_derivs=None):
        """"
        Make a layer computing the log-derivative of a sum layer, required for the EM backwards pass. EM in SPNs can
        be reduced to backprop, see Peharz et al., On the latent variable interpretation in sum-product networks,
        TPAMI 2017.

        The backprop signal at a sum node is simply the sum of backprop signals from all its product parents, times
        the values of its sibling nodes.

        This function insert the required computational ops to do this for all distributions (sums, input distributions)
        in all regions.
        """

        if ps_count >= 0:
            scope_string = 'deriv_sums{}'.format(ps_count)
        else:
            scope_string = 'deriv_dist'

        with tf.variable_scope(scope_string) as scope:
            for region_counter, region in enumerate(regions):
                with tf.variable_scope(str(region_counter)) as scope:
                    dist = self.region_distributions[region]

                    # if we are at the root region, init backprop with 1 (0 in log-domain), or externally provided
                    # derivatives.
                    if region not in self.parent_products:
                        if root_log_derivs is not None:
                            self.log_derivs[dist] = root_log_derivs
                        else:
                            self.log_derivs[dist] = tf.zeros_like(self.region_distributions[region])
                    else:
                        backprop_terms = []

                        for prods in self.parent_products[region]:
                            reshaped_prod_deriv = self.reshaped_prod_derivs[prods]

                            with tf.variable_scope(self.prod_names[prods]) as scope:
                                sibling, expand_dim = self.sibling_dicts[region][prods]
                                if expand_dim == 1:
                                    deriv_factor = tf.reduce_logsumexp(reshaped_prod_deriv + sibling, axis=2)
                                else:
                                    deriv_factor = tf.reduce_logsumexp(reshaped_prod_deriv + sibling, axis=1)

                                backprop_terms.append(deriv_factor)

                        self.log_derivs[dist] = tf.reduce_logsumexp(tf.stack(backprop_terms, axis=0), axis=0)

    def _make_objective_ops(self):
        """Make objective ops, i.e. cross-entropy, log-likelihood, ..."""

        ### labels to be fed in training/evaluation
        self.labels = tf.placeholder(tf.int32, shape=None)

        ### cross-entropy
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.outputs,
            name='xentropy')
        self.cross_entropy_mean = tf.reduce_mean(self.cross_entropy, name='xentropy_mean')

        # add cross-entropy to loss, if trade-off factor (lambda_discriminative) requires it
        if (self.lambda_discriminative > 0.0):
            if (self.kappa_discriminative > 0.0):
                tf.add_to_collection('losses', self.lambda_discriminative * self.kappa_discriminative * self.cross_entropy_mean)

        ### log-likelihood
        range_op = tf.range(tf.squeeze(self.dynamic_batch_size))
        # indicators to the spn outputs corresponding to the labels
        indicators = tf.transpose(tf.stack([range_op, self.labels]))
        self.log_likelihood = tf.gather_nd(self.outputs, indicators)
        self.log_likelihood_mean = tf.reduce_mean(self.log_likelihood)
        self.neg_norm_ll = -self.log_likelihood_mean / self.num_dims
        if self.lambda_discriminative < 1.0:
            tf.add_to_collection('losses', (1. - self.lambda_discriminative) * self.neg_norm_ll)

        ### margin
        competitor_mask = tf.one_hot(self.labels, self.outputs.shape[1], on_value=-np.inf, off_value=0.0)
        competitor_mask.set_shape(self.outputs.get_shape())
        competitors = self.outputs + competitor_mask
        best_competitor_ll = tf.reduce_max(competitors, 1)

        self.log_margin = self.log_likelihood - best_competitor_ll
        self.log_margin_hinged = tf.minimum(self.log_margin, 1.0)
        self.neg_margin_objective = -tf.reduce_mean(self.log_margin_hinged)
        if self.lambda_discriminative > 0.0:
            if self.kappa_discriminative < 1.0:
                tf.add_to_collection('losses', self.lambda_discriminative * (1. - self.kappa_discriminative) * self.neg_margin_objective)

        ### the overal objective
        self.objective = tf.add_n(tf.get_collection('losses'), name='objective')

        ### counting correctly classified examples
        prediction = tf.argmax(self.outputs, axis=1)
        equality = tf.equal(prediction, tf.cast(self.labels, tf.int64))
        self.num_correct = tf.reduce_sum(tf.cast(equality, tf.float32))

    def _make_optimizer_ops(self):
        """Make optimizer ops."""

        if self.optimizer != "none":
            self.learning_rate = tf.get_variable(
                "learning_rate",
                dtype=tf.float32,
                initializer=tf.constant(self.provided_learning_rate))
        else:
            self.learning_rate = None

        if self.optimizer == "adam":
            self.decrease_lr_op = tf.no_op()
            self.optimizer_op = tf.train.AdamOptimizer(learning_rate=self.provided_learning_rate)
        elif self.optimizer == "momentum":
            self.decrease_lr_op = tf.assign(self.learning_rate, self.learning_rate * self.learning_rate_decay)
            self.optimizer_op = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
        elif self.optimizer == "none":
            self.decrease_lr_op = tf.no_op()
            self.optimizer_op = tf.no_op()
        else:
            raise NotImplementedError("Optimizer not implemented.")

        if self.optimizer != "none":
            self.train_op = self.optimizer_op.minimize(self.objective)
        else:
            tf.no_op()

    def _make_EM(self):
        """Make the ops required for EM"""

        rg_layers = self.region_graph_layers

        log_likelihood_ext = tf.reshape(self.log_likelihood, [-1, 1, 1])

        with tf.variable_scope('EM') as scope:

            all_accums = []
            all_accum_updaters = []
            all_weight_updaters = []

            ps_count = 0
            for layerIdx in range(1, len(rg_layers)):
                if layerIdx % 2 == 1:
                    continue

                with tf.variable_scope('sums{}'.format(ps_count)) as scope:

                    for region_counter, region in enumerate(rg_layers[layerIdx]):
                        with tf.variable_scope(str(region_counter)) as scope:
                            sums = self.region_distributions[region]
                            weighted_children = self.weighted_children[sums]
                            log_derivs = tf.expand_dims(self.log_derivs[sums], 1)

                            increment = tf.subtract(log_derivs + weighted_children, log_likelihood_ext, name="increment")
                            accumulator = tf.get_variable('accum',
                                                          initializer=tf.zeros(self.sum_params[sums].shape),
                                                          dtype=tf.float32)

                            # n <- n + inc
                            new_accum = tf.reduce_logsumexp(tf.concat([accumulator, increment], 0), 0, keepdims=True,
                                                            name="new_accum")
                            with tf.control_dependencies([new_accum]):
                                accum_updater = tf.assign(accumulator, new_accum, name="update_accum")

                            lse = tf.expand_dims(tf.reduce_logsumexp(accumulator, 1, name="lse"), 1, name="lse_expnd")
                            weight_updater = tf.assign(self.sum_params[sums], accumulator - lse, name="update_weights")

                            all_accums.append(accumulator)
                            all_accum_updaters.append(accum_updater)
                            all_weight_updaters.append(weight_updater)

                ps_count = ps_count + 1

            log_likelihood_ext2 = tf.reshape(self.log_likelihood, [-1, 1])

            if self.discrete_leaves:
                with tf.variable_scope('discrete') as scope:

                    for region_counter, region in enumerate(rg_layers[0]):
                        with tf.variable_scope(str(region_counter)) as scope:
                            local_inputs = self.local_inputs[region]
                            distributions = self.region_distributions[region]
                            log_derivs = self.log_derivs[distributions]
                            betas = self.discrete_params[distributions]

                            beta_accumulator = tf.get_variable('beta_accum',
                                                               initializer=tf.zeros(betas.shape),
                                                               dtype=tf.float32)

                            p = tf.subtract(log_derivs + distributions, log_likelihood_ext2, name="log_p")
                            p = tf.reshape(p, [-1, 1, 1, self.num_input_distributions])

                            indicators = tf.one_hot(local_inputs, depth=self.num_states, on_value=0.0,
                                                    off_value=-np.inf)
                            indicators = tf.expand_dims(indicators, -1)

                            inc = p + indicators

                            new_accum = tf.reduce_logsumexp(tf.concat([inc, tf.expand_dims(beta_accumulator, 0)], 0), 0)
                            with tf.control_dependencies([new_accum]):
                                accum_updater = tf.assign(beta_accumulator, new_accum, name="update_accum")

                            lse = tf.reduce_logsumexp(beta_accumulator, 1, name="lse", keepdims=True)
                            weight_updater = tf.assign(betas, beta_accumulator - lse, name="update_weights")

                            all_accums.append(beta_accumulator)
                            all_accum_updaters.append(accum_updater)
                            all_weight_updaters.append(weight_updater)
            else:
                with tf.variable_scope('gauss') as scope:

                    for region_counter, region in enumerate(rg_layers[0]):
                        with tf.variable_scope(str(region_counter)) as scope:

                            local_inputs = self.local_inputs[region]
                            gauss = self.region_distributions[region]
                            log_derivs = self.log_derivs[gauss]
                            means = self.gauss_mean_params[gauss]
                            if self.gauss_min_var < self.gauss_max_var:
                                var_params = self.gauss_var_params[gauss]

                            mean_accumulator = tf.get_variable('mean_accum',
                                                               initializer=tf.zeros(means.shape),
                                                               dtype=tf.float32)

                            if self.gauss_min_var < self.gauss_max_var:
                                var_accumulator = tf.get_variable('var_accum',
                                                                  initializer=tf.zeros(means.shape),
                                                                  dtype=tf.float32)

                            lse_p_accumulator = tf.get_variable('norm_p_accum',
                                                                initializer=tf.constant(
                                                                    -np.inf,
                                                                    shape=[1, 1, means.shape[2]],
                                                                    dtype=tf.float32),
                                                                dtype=tf.float32)

                            p = tf.subtract(log_derivs + gauss, log_likelihood_ext2, name="log_p")
                            p = tf.expand_dims(p, 1)
                            lse_p = tf.reduce_logsumexp(p, 0, keepdims=True)
                            norm_p = tf.exp(p - lse_p, name="norm_p")

                            local_inputs = tf.expand_dims(local_inputs, 2)
                            mu_inc = tf.reduce_sum(local_inputs * norm_p, 0, keepdims=True, name="mu_inc")

                            if self.gauss_min_var < self.gauss_max_var:
                                local_inputs_squared = local_inputs ** 2
                                var_inc = tf.reduce_sum(local_inputs_squared * norm_p, 0, keepdims=True,
                                                        name="var_inc")

                            new_lse_p = tf.reduce_logsumexp(tf.concat([lse_p_accumulator, lse_p], 0), axis=0,
                                                            keepdims=True)

                            weighted_mean_accum = mean_accumulator * tf.exp(lse_p_accumulator - new_lse_p)
                            weighted_mean_inc = mu_inc * tf.exp(lse_p - new_lse_p)
                            new_mean = weighted_mean_accum + weighted_mean_inc

                            control_deps = [new_mean, new_lse_p]

                            if self.gauss_min_var < self.gauss_max_var:
                                weighted_var_accum = var_accumulator * tf.exp(lse_p_accumulator - new_lse_p)
                                weighted_var_inc = var_inc * tf.exp(lse_p - new_lse_p)
                                new_var = weighted_var_accum + weighted_var_inc
                                control_deps.append(new_var)

                            with tf.control_dependencies(control_deps):
                                mean_accum_updater = tf.assign(mean_accumulator, new_mean, name="update_mean_accum")
                                p_accum_updater = tf.assign(lse_p_accumulator, new_lse_p, name="update_p_accum")
                                if self.gauss_min_var < self.gauss_max_var:
                                    var_accum_updater = tf.assign(var_accumulator, new_var, name="update_var_accum")

                            mean_updater = tf.assign(means, mean_accumulator, name="update_means")

                            if self.gauss_min_var < self.gauss_max_var:
                                # this is the actual new variance -- we just need to map it back to our parametrization
                                expected_var = var_accumulator - mean_accumulator ** 2
                                if self.gauss_isotropic:
                                    expected_var = tf.reduce_mean(expected_var, axis=1, keep_dims=True)
                                log_expect_var = tf.log(expected_var)

                                min_log_var = np.log(self.gauss_min_var)
                                max_log_var = np.log(self.gauss_max_var)

                                clip_log_expect_var = tf.clip_by_value(log_expect_var, min_log_var + 1e-6, max_log_var - 1e-6)
                                norm_clip_ev = (clip_log_expect_var - min_log_var) / (max_log_var - min_log_var)

                                # inverse sigmoid: -log( 1/x - 1 )
                                var_updater = tf.assign(var_params, -tf.log(1. / norm_clip_ev - 1.))

                            all_accums.append(mean_accumulator)
                            all_accums.append(lse_p_accumulator)
                            if self.gauss_min_var < self.gauss_max_var:
                                all_accums.append(var_accumulator)

                            all_accum_updaters.append(mean_accum_updater)
                            all_accum_updaters.append(p_accum_updater)
                            if self.gauss_min_var < self.gauss_max_var:
                                all_accum_updaters.append(var_accum_updater)

                            all_weight_updaters.append(mean_updater)
                            if self.gauss_min_var < self.gauss_max_var:
                                all_weight_updaters.append(var_updater)

            self.em_reset_accums = tf.variables_initializer(all_accums, "reset_acccums")
            self.em_update_accums = tf.tuple(all_accum_updaters, "update_accums")
            self.em_update_params = tf.tuple(all_weight_updaters, "update_params")

    def _make_spn(self):
        """
        Compile the SPN from the region graph.
        """

        rg_layers = self.region_graph_layers
        if len(rg_layers) == 1 and self.optimizer.lower() != "adam" and self.optimizer.lower() != "none":
            raise AssertionError("Split depth 0 only implemented for adam")

        # make inputs
        with tf.variable_scope('inputs') as scope:
            if self.provided_inputs is None:
                self.inputs = tf.placeholder(tf.float32, shape=(None, self.num_dims), name='inputs')
            else:
                self.inputs = self.provided_inputs

            if self.dropout_rate_input is not None:
                self.dropout_input_placeholder = tf.placeholder(tf.float32, shape=[])
                dropout_shape = tf.shape(self.inputs)
                random_tensor = random_ops.random_uniform(dropout_shape, dtype=tf.float32)
                self.dropout_mask_input = math_ops.floor(self.dropout_input_placeholder + random_tensor, name='dropout_mask')

        # the unknown batch size is sometimes required
        self.dynamic_batch_size = tf.slice(tf.shape(self.inputs), [0], [1])

        if self.discrete_leaves:
            self._make_discrete_layer(rg_layers[0])
        else:
            self._make_gauss_layer(rg_layers[0])

        if len(rg_layers) == 1:
            region = rg_layers[0][0]
            self.region_products[region] = [self.region_distributions[region]]
            self._make_sum_layer(0, rg_layers[0], self.num_classes, None)
        else:
            # make sum-product layers
            if self.dropout_rate_sums is not None:
                self.dropout_sums_placeholder = tf.placeholder(tf.float32, shape=[])

            # alternate between sum and product layers
            ps_count = 0
            for layerIdx in range(1, len(rg_layers)):
                if layerIdx % 2 == 1:
                    self._make_prod_layer(ps_count, rg_layers[layerIdx])
                else:
                    if layerIdx == len(rg_layers) - 1:
                        cur_num_sums = self.num_classes
                        cur_dropout = None
                    else:
                        cur_num_sums = self.num_sums
                        cur_dropout = self.dropout_sums_placeholder
                    self._make_sum_layer(ps_count, rg_layers[layerIdx], cur_num_sums, cur_dropout)
                    ps_count = ps_count + 1

        self.outputs = self.region_distributions[self.root_region]
        self._make_objective_ops()

        if self.optimizer.lower() != "em" and self.optimizer.lower() != "hard_em" :
            self._make_optimizer_ops()
        else:
            # make derivative layers
            ps_count = ps_count - 1
            for layerIdx in range(len(rg_layers) - 1, 0, -1):
                if layerIdx % 2 == 0:
                    if layerIdx == len(rg_layers) - 1:
                        if len(rg_layers[layerIdx]) != 1:
                            raise AssertionError("#root regions != 1")

                        root_region = rg_layers[layerIdx][0]
                        root_distributions = self.region_distributions[root_region]
                        self.EM_deriv_input_pl = tf.placeholder(tf.float32, root_distributions.shape)
                        self._make_dist_backprop_layer(ps_count, rg_layers[layerIdx], self.EM_deriv_input_pl)
                    else:
                        self._make_dist_backprop_layer(ps_count, rg_layers[layerIdx])
                else:
                    self._make_prod_backprop_layer(ps_count, rg_layers[layerIdx + 1])
                    ps_count = ps_count - 1

            self._make_dist_backprop_layer(ps_count, rg_layers[0])
            self._make_EM()
