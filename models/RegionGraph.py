import numpy as np
# from graphviz import Digraph


class RegionGraph(object):
    """
    Class implementing a region graph.

    A region graph is defined w.r.t. a set of indices of random variables in an SPN. E.g., when constructing a region
    graph with a tuple of indices=(0,1,2,3,4,5,6,7), we will end up with a model over 8 random variables,
    X_0, ..., X_7. Indices are typically int's.

    A *region* R is defined as a non-empty subset of the indices, and represented as sorted tuples with unique entries.
    E.g., when indices=(0,1,2,3,4,5,6,7), then (0,1,2,3), (4,5,6,7), (0,7), (1,), (0,1,2,3,4,5,6,7) are examples of
    regions.
    (4,5,6,7,8) is *not* a region (contains 8, which is not in indices).
    (0,1,2,4,3) is *not* a region (not sorted).
    (,) is *not* a region (empty).
    (0,1,2,2,3) is *not* a region (duplicate 2).
    In particular note, that indices, (0,1,2,3,4,5,6,7), itself is a region. It will always be in the region graph, and
    is called the root region.

    A *partition* P of a region R is defined following the usual mathematical definition, i.e., a collection of
    non-empty sets, which are non-overlapping, and whose union is R. Here we represent partitions as sorted tuples of
    regions (which are themselves sorted tuples of elements from indices, see above).
    For region R=(0,1,2,3,4), ((0,1,2), (3,4)), ((0,4), (1,2,3)), and ((0,1,2,3,4),) are examples of partitions. The
    latter is a partition containing only R. This is valid, but we will not use these (only proper partitions).
    ((3,4), ((0,1,2)) is *not* a partition (not sorted)
    ((0,1,2), (2,3,4)) is *not* a partition (overlapping 2)
    ((0,1), (2,3)) is *not* a partition (of R=(0,1,2,3,4)) (missing 4)
    R is called the *parent region* of P.
    Any C, such that C in P == True, is called *child region* of P.
    Note that tuple(sorted(e for C in P for e in C)) yields the parent region R.

    Summary so far:
    region = sorted tuple of ints (elements unique, elements only taken from indices)
    partition = sorted tuple of regions (regions non-overlapping, regions non-empty, at least 2 regions)

    A *region graph* is an acyclic, directed, bi-partite graph over regions and partitions, i.e. any child of a region
    R is a partition of R, and any child of a partition is a child region of the partition. The root of the region
    graph is indices (e.g. (0,1,2,3,4,5,6,7) from above). The leaves of the region graph must also be regions. They are
    called input regions, or leaf regions.

    Given a region graph, we can easily construct a corresponding SPN (see RatSpn.py):
    1) Associate I distributions to each input region.
    2) Associate K sum nodes to each other (non-input) region.
    3) For each partition P in the region graph, take all cross-products (product nodes) of distributions/sum nodes
    associated with the child regions. Connect these products as children of all sum nodes in the parent region of P.

    This procedure will always deliver a complete and decomposable SPN.
    """

    def __init__(self, items, seed=12345):

        self._items = tuple(sorted(items))

        # Regions
        self._regions = set()
        self._child_partitions = dict()

        # Partitions
        self._partitions = set()

        # Private random generator
        self._rand_state = np.random.RandomState(seed)

        # layered representation of the region graph
        self._layers = []

        # The root region (== _items) is already part of the region graph
        self._regions.add(self._items)

    def get_root_region(self):
        """Get root region."""
        return self._items

    def get_regions(self):
        """Get set of all regions."""
        return self._regions

    def get_child_partitions(self, region):
        """Get list of all child partitions of regions. """
        return self._child_partitions[region]

    def get_leaf_regions(self):
        """Get list of leaf regions, i.e. regions which don't have child partitions."""
        return [x for x in self._regions if x not in self._child_partitions]

    @staticmethod
    def get_parent_region(partition):
        """Get back the parent region of partition."""
        return tuple(sorted(e for child in partition for e in child))

    def insert_partition(self, partition):
        """
        Insert a partition in the region graph.
        Consistency check: we can only insert a partition of an existing region.
        Do bookkeeping to implement region graph as linked list.
        """

        parent_region = self.get_parent_region(partition)

        # we only allow to include partitions of existing regions
        if parent_region not in self._regions:
            raise AssertionError("Parent region not found.")

        # insert sub_regions
        for sub_region in partition:
            self._regions.add(sub_region)

        if partition not in self._partitions:
            self._partitions.add(partition)
            region_children = self._child_partitions.get(parent_region, [])
            self._child_partitions[parent_region] = region_children + [partition]

    def random_split(self, num_parts, num_recursions=1, region=None):
        """
        Split a region in n random parts and introduce the corresponding partition in the region graph.
        Recursive method.
        """

        if num_recursions < 1:
            return None

        if not region:
            region = self._items

        if region not in self._regions:
            raise LookupError('Trying to split non-existing region.')

        if len(region) == 1:
            return None

        def chunk_list(region_list, num_parts):
            q = len(region_list) // num_parts
            r = len(region_list) % num_parts

            chunks = []
            idx = 0
            for k in range(0, num_parts):
                inc = q + 1 if k < r else q
                sub_list = region_list[idx:idx + inc]
                chunks.append(sub_list)
                idx = idx + inc
            return chunks

        permuted_region = list(self._rand_state.permutation(list(region)))
        partition = chunk_list(permuted_region, min(len(permuted_region), num_parts))
        partition = [tuple(sorted(x)) for x in partition]
        partition = tuple(sorted(partition))
        self.insert_partition(partition)

        if num_recursions > 1:
            for r in partition:
                self.random_split(num_parts, num_recursions-1, r)

        return partition

    def make_layers(self):
        """
        Make a layered structure, represented as a list of lists.

        _layer[0] will contain leaf regions
        For k > 0, the layers will alternate, i.e.
            _layer[k] will contain partitions, if k is odd
            _layer[k] will contain regions, if k is even

        This layered representation is greedily constructed, in order to contain as few layers as possible. Crucially,
        it respects some topological order of the region graph (a directed graph is acyclic if and only if there
        exists a topological order of its nodes), i.e. if k >= l, it is guaranteed that regions (partitions) in layer k
        cannot be children of partitions (regions) in layer l.
        """

        seen_regions = set()
        seen_partitions = set()

        leaf_regions = self.get_leaf_regions()
        # sort regions lexicographically
        leaf_regions = [tuple(sorted(i)) for i in sorted([sorted(j) for j in leaf_regions])]
        self._layers = [leaf_regions]
        if (len(leaf_regions) == 1) and (self._items in leaf_regions):
            return self._layers

        seen_regions.update(leaf_regions)

        while len(seen_regions) != len(self._regions) or len(seen_partitions) != len(self._partitions):
            # the next partition layer contains all partitions which have not been visited (seen)
            # and all its child regions have been visited
            next_partition_layer = [p for p in self._partitions if p not in seen_partitions
                                    and all([r in seen_regions for r in p])]
            self._layers.append(next_partition_layer)
            seen_partitions.update(next_partition_layer)

            # similar as above, but now for regions
            next_region_layer = [r for r in self._regions if r not in seen_regions
                                 and all([p in seen_partitions for p in self._child_partitions[r]])]
            # sort regions lexicographically
            next_region_layer = [tuple(sorted(i)) for i in sorted([sorted(j) for j in next_region_layer])]

            self._layers.append(next_region_layer)
            seen_regions.update(next_region_layer)

        return self._layers

    def make_poon_structure(self, width, height, delta, max_split_depth=None):
        """
        Make a Poon & Domingos like region graph.

        :param width: image width
        :param height: image height
        :param delta: split step-size
        :param max_split_depth: stop splitting at this depth
        :return:
        """

        if self._items != tuple(range(width * height)):
            raise AssertionError('Item set needs to be tuple(range(width * height)).')

        if type(delta) != int or delta <= 0:
            raise AssertionError('delta needs to be a nonnegative integer.')

        def split(A, axis_idx, x):
            """This splits a multi-dimensional numpy array in one axis, at index x.
            For example, if A =
            [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]

            then split(A, 0, 1) delivers
            [[1, 2, 3, 4]],

            [[5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]
            """
            slc = [slice(None)] * len(A.shape)
            slc[axis_idx] = slice(0, x)
            A1 = A[tuple(slc)]
            slc[axis_idx] = slice(x, A.shape[axis_idx])
            A2 = A[tuple(slc)]
            return A1, A2

        img = np.reshape(range(height * width), (height, width))
        img_tuple = tuple(sorted(img.reshape(-1)))

        # Q is a queue
        Q = [img]
        depth_dict = {img_tuple: 0}

        while Q:
            region = Q.pop(0)
            region_tuple = tuple(sorted(region.reshape(-1)))
            depth = depth_dict[region_tuple]
            if max_split_depth is not None and depth >= max_split_depth:
                continue

            region_children = []

            for axis, length in enumerate(region.shape):
                if length <= delta:
                    continue

                num_splits = int(np.ceil(length / delta) - 1)
                split_points = [(x + 1) * delta for x in range(num_splits)]

                for idx in split_points:
                    region_1, region_2 = split(region, axis, idx)

                    region_1_tuple = tuple(sorted(region_1.reshape(-1)))
                    region_2_tuple = tuple(sorted(region_2.reshape(-1)))

                    if region_1_tuple not in self._regions:
                        self._regions.add(region_1_tuple)
                        depth_dict[region_1_tuple] = depth + 1
                        Q.append(region_1)

                    if region_2_tuple not in self._regions:
                        self._regions.add(region_2_tuple)
                        depth_dict[region_2_tuple] = depth + 1
                        Q.append(region_2)

                    partition = tuple(sorted([region_1_tuple, region_2_tuple]))

                    if partition in self._partitions:
                        raise AssertionError('Partition already generated -- this should not happen.')

                    self._partitions.add(partition)
                    region_children.append(partition)

            if region_children:
                self._child_partitions[region_tuple] = region_children

#    def render_dot(self, path):
#
#        region_to_label = {}
#        partition_to_label = {}
#
#        dot = Digraph()
#
#        for counter, region in enumerate(self._regions):
#            label = 'R' + str(counter)
#            dot.node(label, label=str(list(region)))
#            region_to_label[region] = label
#
#        dot.attr('node', shape='box')
#        for counter, partition in enumerate(self._partitions):
#            label = 'P' + str(counter)
#            dot.node(label, label='X')
#            partition_to_label[partition] = label
#
#        for region in self._regions:
#            if region not in self._child_partitions:
#                continue
#            for partition in self._child_partitions[region]:
#                dot.edge(region_to_label[region], partition_to_label[partition])
#
#        for partition in self._partitions:
#            for region in partition:
#                dot.edge(partition_to_label[partition], region_to_label[region])
#
#        dot.render(path)

if __name__ == '__main__':

    rg = RegionGraph([1, 2, 3, 4, 5, 6, 7])
    for k in range(3):
        rg.random_split(2, 2)
    layers = rg.make_layers()

    for k in reversed(layers):
        print(k)
