from __future__ import annotations

from typing import Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, lax, random
from jaxtyping import Bool, Int, UInt


class ClusterSolution(eqx.Module):
    """
    A traditional cluster solver found in many Wolff and Swendsen-Wang papers
    does not work within a JAX context as the shape of neighbours in a cluster
    cannot be known at compile time and thus cannot be probed using array
    programming approaches.

    This forces us to come up with a fully vectorised clustering solution,
    which incidentally is also very efficient.

    While the code below looks tiny, it took an awfully long time to come
    up with this solution after trying less vectorised versions that relied
    on a large number of LAX primitives.

    Additionally this solves the problem of multiple visits which many iterative
    approaches neglect. If not mitigated against, this leads to linking rates
    that are too high, especially in higher dimensional spaces.
    """

    rng_key: RNGKey
    links: Bool[Array, "ndim, *dims"]

    @classmethod
    def clusterise_state(cls, rng_key: RNGKey, state: State) -> Self:
        spins = state.spins
        shape = spins.shape

        # Construct an array where the first axis holds the different layers
        # Each layer corresponds to neighbours along a particular axis
        neighbours = np.empty((spins.ndim, *shape), dtype=spins.dtype)
        for i in range(spins.ndim):
            neighbours[i] = jnp.roll(spins, shift=-1, axis=i)

        # Compute a link factor for each neighbour site
        link_factors = get_cluster_linkage_factors(
            state=state, spins=spins, neighbours=neighbours
        )

        # Threshold value that our random number must be below for us to
        # establish a link
        link_thresholds = 1.0 - jnp.exp(link_factors)

        # Generate a random number for each site.
        # In Wolff algorithm where only a single cluster is flipped it might
        # seem a bit excessive to generate this many numbers since PRNG generation
        # is expensive.
        # It has to be done this way though as the number of sites in a cluster
        # cannot be known and we require known shapes at compile time
        # Generating unused keys is a small price compared to the efficiency of
        # doing the clustering in a vectorised manner.
        randoms = random.uniform(key=rng_key, shape=link_thresholds.shape)

        # Establish all links
        # After this, our spin-link graph is fully computed
        # Selecting the generated clusters is a difficult problem in its own
        # right in a vectorised context. This responsibility is given to
        # the ClusterSelection class.
        links = link_thresholds > randoms

        return ClusterSolution(rng_key=rng_key, links=links)


class ClusterSelection(eqx.Module):
    selected: Bool[Array, "*dims"]
    cluster_solution: ClusterSolution
    is_done: bool
    steps: int
    """
    Marks a single selected cluster.

    This is done by iteratively following and marking sites from an initial
    seed site via the links obtained in the cluster solution.

    Together `cluster_solution.links` and `selected` make up a graph of spin
    sites and interconnecting links.

    This process is fundamentally unparallelisable and cannot be vectorised
    further than what is done below (after for loop is unrolled at compile time).

    Attributes:
        selected: Array of bools masking which spin sites are part of this
            specific cluster.
        cluster_solution: A solution from the clustering routine.
            This object holds the probabilistically determined links
        is_done: boolean flag demarking whether cluster selection has terminated
        steps: number of steps it has taken to find cluster from seed index
        """

    @classmethod
    def new(
        cls, selected: Bool[Array, *dims], cluster_solution: ClusterSolution
    ) -> Self:
        return cls(
            selected=selected,
            cluster_solution=cluster_solution,
            is_done=False,
            steps=0,
        )

    @classmethod
    def from_seed_idx(
        cls, cluster_solution: ClusterSolution, seed_idx: UInt[Array, a]
    ) -> Self:
        """
        Find a full selection from a single seed site.

        Uses a un-unrollable while-loop underneath.
        """
        spins_shape = cluster_solution.links.shape[1:]
        selected = jnp.zeros(shape=spins_shape, dtype=bool)
        selected = selected.at[tuple(seed_idx)].set(True)

        selector = cls.new(selected=selected, cluster_solution=cluster_solution)
        selector = lax.while_loop(
            selector.should_continue, selector.expand_selection_step, selector
        )

        return selector

    @staticmethod
    def should_continue(cluster_selector: ClusterSelection) -> bool:
        """
        Used as part of the LAX `while_loop` to determine whether to exit loop.
        """
        return cluster_selector.is_done == False  # noqa: E712

    @staticmethod
    def expand_selection_step(cluster_selector: ClusterSelection) -> ClusterSelection:
        """
        A single step that expands our cluster selection by following the
        edges in our spin-link graph.
        """
        selector = cluster_selector
        solution = cluster_selector.cluster_solution
        # Holds all selected sites this round
        selected = selector.selected

        # This can be possibly be done with broadcasting,
        # but since we JIT anyway we prefer a more readable version.
        # Since selection shape is known at compile time this loop gets
        # unrolled trivially by XLA.
        for i in range(selector.selected.ndim):
            # i denotes the 'layer', where each layer corresponds to
            # links along a particular axis.
            # In 3D: [0] is the up-down axis, [1] is left-right, [2] is in-out
            # Below we use up/down notation for all axis though

            # All links connected to selected sites
            links_to_try = selector.selected | jnp.roll(selector.selected, -1, i)

            # Filter: only the links we have activated
            links_selected = links_to_try & solution.links[i]

            # Expand selected links since links are bidirectional
            links_selected = links_selected | jnp.roll(links_selected, 1, i)

            # Add newly selected sites
            selected = selected | links_selected

        # The sites that changed since last iteration
        new_sites = selected ^ selector.selected
        is_done = new_sites.sum() == 0

        return ClusterSelection(
            selected=selected,
            cluster_solution=solution,
            is_done=is_done,
            steps=selector.steps + 1,
        )


class ClusterSolverState(eqx.Module):
    """
    A previous attempt at an iterative cluster grower.

    There are a few bugs left, but the main principle is working.
    Since this does selection and clustering in the same loop, this
    algorithm is inefficient for evolvers that flip more than a single
    cluster (Swendsen-Wang), but work decently for single-cluster methods
    (Wolff).

    It uses a convolution behind the scenes to expand the selected cluster.
    """

    do_not_visit: Bool[Array, ...]
    updated: Bool[Array, ...]

    neighbour_kernel: Int[Array, ...]

    rng_key: RNGKey
    threshold: float

    steps: int
    is_done: bool

    @staticmethod
    def should_continue(cluster_state: Self) -> bool:
        return cluster_state.is_done == False  # noqa: E712

    @staticmethod
    def evolve(cluster_state: Self) -> Self:
        cs = cluster_state
        new_key, randoms_key = random.split(cluster_state.rng_key)

        # Find neighbours
        neighbour_finder = cs.updated.astype(int)
        neighbour_finder = convolve_with_wrapping(neighbour_finder, cs.neighbour_kernel)
        neighbours = jnp.clip(neighbour_finder, 0, 2).astype(bool)

        # Find candidates
        candidates = neighbours & ~cs.do_not_visit & ~cs.updated

        # Generate random numbers for probabilistic addition to cluster
        randoms = random.uniform(key=randoms_key, shape=candidates.shape)
        random_accept = cs.threshold > randoms

        # Find new cluster members
        new_cluster_members = candidates & random_accept

        # Reflect updates on other arrays
        updated = cs.updated | new_cluster_members

        # We have visited all states this time around, thus
        # no unvisited states if we didn't add new members
        # in which case we are done
        is_done = new_cluster_members.sum() == 0

        return ClusterSolverState(
            do_not_visit=cs.do_not_visit,
            updated=updated,
            neighbour_kernel=cs.neighbour_kernel,
            rng_key=new_key,
            threshold=cs.threshold,
            steps=cs.steps + new_cluster_members.sum(),
            is_done=is_done,
        )
