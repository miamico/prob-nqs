from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import jax
import numpy as np
from jax import numpy as jnp
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.pgm import AbstractNode

from pcnqs.sampling.backend import (
    ClampedSamples,
    DbmSamplingModel,
    FrbmSamplingModel,
    JointSamples,
    SamplingBackend,
)
from pcnqs.types import SpinArray, SpinBatch

FreeSuperBlocks: TypeAlias = list[tuple[Block[AbstractNode], ...] | Block[AbstractNode]]


@dataclass(frozen=True)
class _FrbmProgram:
    ebm: IsingEBM
    program: IsingSamplingProgram
    visible_block: Block[AbstractNode]
    hidden_block: Block[AbstractNode]
    free_blocks: list[Block[AbstractNode]]


@dataclass(frozen=True)
class _DbmProgram:
    ebm: IsingEBM
    free_program: IsingSamplingProgram
    clamped_program: IsingSamplingProgram
    visible_block: Block[AbstractNode]
    hidden_block: Block[AbstractNode]
    deep_block: Block[AbstractNode]
    free_blocks: list[Block[AbstractNode]]
    clamped_free_blocks: list[Block[AbstractNode]]


def _new_spin_node() -> AbstractNode:
    return cast(AbstractNode, SpinNode())  # type: ignore[no-untyped-call]


def _spins_to_bool(spins: SpinBatch | SpinArray) -> np.ndarray:
    """Convert {-1, +1} spins to THRML bool state representation."""

    arr = np.asarray(spins)
    if not np.all(np.isin(arr, [-1, 1])):
        raise ValueError("spin arrays must contain only values in {-1, +1}")
    return (arr > 0).astype(np.bool_)


def _bool_to_spins(state: np.ndarray) -> SpinBatch:
    """Convert THRML bool states back to {-1, +1} spins."""

    return (state.astype(np.int8) * np.int8(2) - np.int8(1)).astype(np.int8)


def _squeeze_chain_axis(samples: np.ndarray) -> np.ndarray:
    if samples.ndim >= 3 and samples.shape[1] == 1:
        return samples[:, 0, ...]
    return samples


def _empty_batch_shape() -> tuple[int]:
    return cast(tuple[int], ())


class ThrmlSamplingBackend(SamplingBackend):
    """THRML/JAX implementation of the PC sampler boundary."""

    def sample_free(
        self,
        model: FrbmSamplingModel | DbmSamplingModel,
        n_samples: int,
        burn_in: int,
        thin: int,
        seed: int,
    ) -> JointSamples:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if thin < 1:
            raise ValueError("thin must be >= 1")

        if isinstance(model, FrbmSamplingModel):
            return self._sample_free_frbm(
                model=model,
                n_samples=n_samples,
                burn_in=burn_in,
                thin=thin,
                seed=seed,
            )
        return self._sample_free_dbm(
            model=model,
            n_samples=n_samples,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
        )

    def sample_clamped(
        self,
        model: DbmSamplingModel,
        clamp_v: SpinBatch,
        n_steps: int,
        seed: int,
    ) -> list[ClampedSamples]:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if clamp_v.ndim != 2:
            raise ValueError("clamp_v must have shape (n_configs, n_visible)")

        program = self._build_dbm_program(model)
        schedule = SamplingSchedule(n_warmup=0, n_samples=n_steps, steps_per_sample=1)

        root_key = jax.random.PRNGKey(seed)
        keys = jax.random.split(root_key, 2 * clamp_v.shape[0] + 1)

        out: list[ClampedSamples] = []
        for idx in range(clamp_v.shape[0]):
            init_key = keys[2 * idx]
            sample_key = keys[2 * idx + 1]

            init_state = hinton_init(
                init_key,
                program.ebm,
                program.clamped_free_blocks,
                batch_shape=_empty_batch_shape(),
            )
            clamped_state = [_spins_to_bool(clamp_v[idx])]

            sampled = sample_states(
                sample_key,
                program.clamped_program,
                schedule,
                init_state_free=init_state,
                state_clamp=clamped_state,
                nodes_to_sample=[program.hidden_block, program.deep_block],
            )

            hidden = _bool_to_spins(_squeeze_chain_axis(np.asarray(sampled[0], dtype=np.bool_)))
            deep = _bool_to_spins(_squeeze_chain_axis(np.asarray(sampled[1], dtype=np.bool_)))
            out.append(ClampedSamples(hidden=hidden, deep=deep))

        return out

    def _sample_free_frbm(
        self,
        model: FrbmSamplingModel,
        n_samples: int,
        burn_in: int,
        thin: int,
        seed: int,
    ) -> JointSamples:
        compiled = self._build_frbm_program(model)
        schedule = SamplingSchedule(n_warmup=burn_in, n_samples=n_samples, steps_per_sample=thin)

        key = jax.random.PRNGKey(seed)
        init_key, sample_key = jax.random.split(key, 2)

        init_state = hinton_init(
            init_key,
            compiled.ebm,
            compiled.free_blocks,
            batch_shape=_empty_batch_shape(),
        )
        sampled = sample_states(
            sample_key,
            compiled.program,
            schedule,
            init_state_free=init_state,
            state_clamp=[],
            nodes_to_sample=[compiled.visible_block, compiled.hidden_block],
        )

        visible = _bool_to_spins(_squeeze_chain_axis(np.asarray(sampled[0], dtype=np.bool_)))
        hidden = _bool_to_spins(_squeeze_chain_axis(np.asarray(sampled[1], dtype=np.bool_)))
        return JointSamples(visible=visible, hidden=hidden, deep=None)

    def _sample_free_dbm(
        self,
        model: DbmSamplingModel,
        n_samples: int,
        burn_in: int,
        thin: int,
        seed: int,
    ) -> JointSamples:
        compiled = self._build_dbm_program(model)
        schedule = SamplingSchedule(n_warmup=burn_in, n_samples=n_samples, steps_per_sample=thin)

        key = jax.random.PRNGKey(seed)
        init_key, sample_key = jax.random.split(key, 2)

        init_state = hinton_init(
            init_key,
            compiled.ebm,
            compiled.free_blocks,
            batch_shape=_empty_batch_shape(),
        )
        sampled = sample_states(
            sample_key,
            compiled.free_program,
            schedule,
            init_state_free=init_state,
            state_clamp=[],
            nodes_to_sample=[compiled.visible_block, compiled.hidden_block, compiled.deep_block],
        )

        visible = _bool_to_spins(_squeeze_chain_axis(np.asarray(sampled[0], dtype=np.bool_)))
        hidden = _bool_to_spins(_squeeze_chain_axis(np.asarray(sampled[1], dtype=np.bool_)))
        deep = _bool_to_spins(_squeeze_chain_axis(np.asarray(sampled[2], dtype=np.bool_)))
        return JointSamples(visible=visible, hidden=hidden, deep=deep)

    def _build_frbm_program(self, model: FrbmSamplingModel) -> _FrbmProgram:
        n_visible = model.visible_biases.shape[0]
        n_hidden = model.hidden_biases.shape[0]

        visible_nodes = [_new_spin_node() for _ in range(n_visible)]
        hidden_nodes = [_new_spin_node() for _ in range(n_hidden)]
        all_nodes = [*visible_nodes, *hidden_nodes]

        edges: list[tuple[AbstractNode, AbstractNode]] = []
        weights: list[float] = []
        for i in range(n_visible):
            for j in range(n_hidden):
                wij = float(model.weights_vh[i, j])
                if wij == 0.0:
                    continue
                edges.append((visible_nodes[i], hidden_nodes[j]))
                weights.append(wij)

        biases = np.concatenate((model.visible_biases, model.hidden_biases)).astype(np.float64)
        weights_array = np.asarray(weights, dtype=np.float64)

        ebm = IsingEBM(
            nodes=all_nodes,
            edges=edges,
            biases=jnp.asarray(biases),
            weights=jnp.asarray(weights_array),
            beta=jnp.asarray(float(model.beta)),
        )

        visible_block = Block(visible_nodes)
        hidden_block = Block(hidden_nodes)
        free_blocks = [visible_block, hidden_block]
        free_super_blocks: FreeSuperBlocks = [visible_block, hidden_block]
        program = IsingSamplingProgram(ebm=ebm, free_blocks=free_super_blocks, clamped_blocks=[])

        return _FrbmProgram(
            ebm=ebm,
            program=program,
            visible_block=visible_block,
            hidden_block=hidden_block,
            free_blocks=free_blocks,
        )

    def _build_dbm_program(self, model: DbmSamplingModel) -> _DbmProgram:
        n_visible = model.visible_biases.shape[0]
        n_hidden = model.hidden_biases.shape[0]
        n_deep = model.deep_biases.shape[0]

        visible_nodes = [_new_spin_node() for _ in range(n_visible)]
        hidden_nodes = [_new_spin_node() for _ in range(n_hidden)]
        deep_nodes = [_new_spin_node() for _ in range(n_deep)]
        all_nodes = [*visible_nodes, *hidden_nodes, *deep_nodes]

        edges: list[tuple[AbstractNode, AbstractNode]] = []
        weights: list[float] = []

        for i in range(n_visible):
            for j in range(n_hidden):
                wij = float(model.weights_vh[i, j])
                if wij == 0.0:
                    continue
                edges.append((visible_nodes[i], hidden_nodes[j]))
                weights.append(wij)

        for j in range(n_hidden):
            for k in range(n_deep):
                wjk = float(model.weights_hd[j, k])
                if wjk == 0.0:
                    continue
                edges.append((hidden_nodes[j], deep_nodes[k]))
                weights.append(wjk)

        biases = np.concatenate(
            (model.visible_biases, model.hidden_biases, model.deep_biases)
        ).astype(np.float64)
        weights_array = np.asarray(weights, dtype=np.float64)

        ebm = IsingEBM(
            nodes=all_nodes,
            edges=edges,
            biases=jnp.asarray(biases),
            weights=jnp.asarray(weights_array),
            beta=jnp.asarray(float(model.beta)),
        )

        visible_block = Block(visible_nodes)
        hidden_block = Block(hidden_nodes)
        deep_block = Block(deep_nodes)

        free_blocks = [visible_block, hidden_block, deep_block]
        free_super_blocks: FreeSuperBlocks = [visible_block, hidden_block, deep_block]
        free_program = IsingSamplingProgram(
            ebm=ebm,
            free_blocks=free_super_blocks,
            clamped_blocks=[],
        )

        clamped_free_blocks = [hidden_block, deep_block]
        clamped_super_blocks: FreeSuperBlocks = [hidden_block, deep_block]
        clamped_program = IsingSamplingProgram(
            ebm=ebm,
            free_blocks=clamped_super_blocks,
            clamped_blocks=[visible_block],
        )

        return _DbmProgram(
            ebm=ebm,
            free_program=free_program,
            clamped_program=clamped_program,
            visible_block=visible_block,
            hidden_block=hidden_block,
            deep_block=deep_block,
            free_blocks=free_blocks,
            clamped_free_blocks=clamped_free_blocks,
        )
