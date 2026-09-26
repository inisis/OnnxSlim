from typing import List, Optional, Tuple

import numpy as np

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern

_PATTERN_TEMPLATE = """
    input      input        0  1  transpose_0
    Transpose  transpose_0  1  1  input view_op
    {view_op_type}  view_op  1+ 1  transpose_0 transpose_1
    Transpose  transpose_1  1  1  view_op output
    output     output       1  0  transpose_1
"""

# Scope and approach follow onnxruntime's transpose optimizer (onnx_transpose_optimization.cc):
# Squeeze/Unsqueeze move one axis at a time (always safe), Reshape only *splits* post-transpose axes


def _split_groups_by_origin(
    transposed_shape: List[int], requested_shape: List[int], perm0: List[int]
) -> Optional[List[Tuple[int, int]]]:
    """Splits `requested_shape` into one contiguous run per `transposed_shape` entry (products must
    match), then reorders the runs into pre-transpose axis order. None if the products don't line up.
    """
    groups: List[Optional[Tuple[int, int]]] = [None] * len(perm0)
    cursor = 0
    n = len(requested_shape)
    for j, target in enumerate(transposed_shape):
        start = cursor
        prod = 1
        while True:
            if cursor >= n:
                return None
            prod *= requested_shape[cursor]
            cursor += 1
            if prod >= target:
                break
        if prod != target:
            return None
        groups[perm0[j]] = (start, cursor)
    return groups if cursor == n else None


def _reshape_as_perm(transposed_shape: List[int], requested_shape: List[int]) -> Optional[List[int]]:
    """Returns `perm` with `requested_shape[i] == transposed_shape[perm[i]]`, when the reshape only
    relocates size-1 axes - the same condition TransposeAsReshapeMatcher checks in reverse.
    """
    src_real = [i for i, d in enumerate(transposed_shape) if d != 1]
    dst_real = [i for i, d in enumerate(requested_shape) if d != 1]
    if [transposed_shape[i] for i in src_real] != [requested_shape[i] for i in dst_real]:
        return None
    src_ones = [i for i, d in enumerate(transposed_shape) if d == 1]
    dst_ones = [i for i, d in enumerate(requested_shape) if d == 1]
    perm = [0] * len(transposed_shape)
    for s, d in zip(src_real, dst_real):
        perm[d] = s
    for s, d in zip(src_ones, dst_ones):
        perm[d] = s
    return perm


class _TransposeViewTransposeMatcherBase(PatternMatcher):
    """Shared matcher for `Transpose -> <ViewOp> -> Transpose`: moves the ViewOp before the first
    Transpose. Subclasses only implement `_compute_rewrite` (moved perm/shape/inputs from `perm0`).
    The trailing Transpose's perm is never read - it's just left for EliminationTranspose to merge next.
    """

    def __init__(self, priority, view_op_type):
        super().__init__(Pattern(_PATTERN_TEMPLATE.format(view_op_type=view_op_type)), priority)

    @property
    def name(self):
        return "EliminationTransposeViewTranspose"

    def parameter_check(self):
        return len(self.transpose_0.users) == 1 and len(self.view_op.users) == 1

    def _compute_rewrite(
        self, view_node: gs.Node, a_shape: List[int], perm0: List[int]
    ) -> Optional[Tuple[List[int], List[int], List[gs.Tensor]]]:
        raise NotImplementedError

    def rewrite(self, opset=11):
        t0 = self.transpose_0
        v = self.view_op

        a = t0.inputs[0]
        a_shape = a.shape

        if a_shape is None or any(not isinstance(d, int) or d <= 0 for d in a_shape):
            return {}

        if "perm" in t0.attrs:
            perm0 = list(t0.attrs["perm"])
        else:
            perm0 = list(reversed(range(len(a_shape))))
        plan = self._compute_rewrite(v, a_shape, perm0)
        if plan is None:
            return {}
        new_perm, new_shape, extra_inputs = plan

        b = t0.outputs[0]
        c = v.outputs[0]

        t0.inputs.clear()
        t0.outputs.clear()
        v.inputs.clear()
        v.outputs.clear()

        v.inputs.append(a)
        v.inputs.extend(extra_inputs)
        v.outputs.append(b)

        t0.inputs.append(b)
        t0.outputs.append(c)
        t0.attrs["perm"] = new_perm

        b.shape = new_shape
        b.dtype = a.dtype

        return {}


@register_fusion_pattern(priority=1, min_opset=5)
class ReshapeViewMatcher(_TransposeViewTransposeMatcherBase):
    """Two cases: equal rank, where the Reshape only relocates size-1 axes (treated as a permutation and
    composed with perm0); and rank-increasing, where it *splits* each post-transpose axis into a
    contiguous run of output axes (ports onnxruntime's HandleReshapeSplit).
    """

    def __init__(self, priority):
        super().__init__(priority, "Reshape")

    def _compute_rewrite(self, view_node, a_shape, perm0):
        if not isinstance(view_node.inputs[1], gs.Constant):
            return None
        requested_shape = [int(d) for d in view_node.inputs[1].values.tolist()]
        if any(d <= 0 for d in requested_shape):
            return None

        rank0 = len(perm0)
        transposed_shape = [a_shape[p] for p in perm0]

        if len(requested_shape) == rank0:
            reshape_perm = _reshape_as_perm(transposed_shape, requested_shape)
            if reshape_perm is None:
                return None
            new_perm = [perm0[p] for p in reshape_perm]
            shape_const = gs.Constant(
                name=f"{view_node.outputs[0].name}_shape",
                values=np.array(a_shape, dtype=np.int64),
            )
            return new_perm, list(a_shape), [shape_const]

        if len(requested_shape) < rank0:
            return None  # merging multiple post-transpose axes together is out of scope

        groups = _split_groups_by_origin(transposed_shape, requested_shape, perm0)
        if groups is None:
            return None

        new_shape = []
        new_perm = [0] * len(requested_shape)
        for start, end in groups:
            for k in range(start, end):
                new_perm[k] = len(new_shape)
                new_shape.append(requested_shape[k])

        shape_const = gs.Constant(
            name=f"{view_node.outputs[0].name}_shape",
            values=np.array(new_shape, dtype=np.int64),
        )
        return new_perm, new_shape, [shape_const]


@register_fusion_pattern(priority=1, min_opset=13)
class UnsqueezeViewMatcher(_TransposeViewTransposeMatcherBase):
    """opset >= 13 only (axes as a Constant input). `axes` refers to *output* positions, so it's reused
    unchanged regardless of perm0 - only the moved Transpose's perm needs recomputing (ORT's UnsqueezePerm).
    """

    def __init__(self, priority):
        super().__init__(priority, "Unsqueeze")

    def _compute_rewrite(self, view_node, a_shape, perm0):
        if not isinstance(view_node.inputs[1], gs.Constant):
            return None
        axes_input = view_node.inputs[1]
        rank0 = len(perm0)
        raw_axes = axes_input.values.tolist()
        k = len(raw_axes)
        new_rank = rank0 + k
        axes = sorted(int(a) + new_rank if a < 0 else int(a) for a in raw_axes)
        if len(set(axes)) != k or any(a < 0 or a >= new_rank for a in axes):
            return None

        is_added = [False] * new_rank
        for a in axes:
            is_added[a] = True
        axes_map = [i for i in range(new_rank) if not is_added[i]]  # old axis -> new position

        new_perm = []
        new_shape = []
        j = a_idx = 0
        for i in range(new_rank):
            if is_added[i]:
                new_perm.append(i)
                new_shape.append(1)
            else:
                new_perm.append(axes_map[perm0[j]])
                j += 1
                new_shape.append(a_shape[a_idx])
                a_idx += 1

        return new_perm, new_shape, [axes_input]


@register_fusion_pattern(priority=1, min_opset=13)
class SqueezeViewMatcher(_TransposeViewTransposeMatcherBase):
    """opset >= 13 only (axes as an optional Constant input). Unlike Unsqueeze, `axes` refers to *input*
    positions, so it must be translated through `perm0` into A's own axis order (ORT's SqueezePerm).
    """

    def __init__(self, priority):
        super().__init__(priority, "Squeeze")

    def _compute_rewrite(self, view_node, a_shape, perm0):
        rank0 = len(perm0)
        if len(view_node.inputs) >= 2:
            if not isinstance(view_node.inputs[1], gs.Constant):
                return None
            axes = [int(a) + rank0 if a < 0 else int(a) for a in view_node.inputs[1].values.tolist()]
        else:
            # axes omitted: squeeze every statically-known size-1 (post-transpose) axis.
            axes = [i for i in range(rank0) if a_shape[perm0[i]] == 1]

        if len(set(axes)) != len(axes) or any(a < 0 or a >= rank0 for a in axes):
            return None
        if any(a_shape[perm0[a]] != 1 for a in axes):
            return None

        removed = {perm0[a] for a in axes}  # translate to A's own axis indices

        axes_map = {}
        j = 0
        for i in range(rank0):
            if i not in removed:
                axes_map[i] = j
                j += 1
        new_perm = [axes_map[p] for p in perm0 if p not in removed]
        new_shape = [a_shape[i] for i in range(rank0) if i not in removed]

        axes_const = gs.Constant(
            name=f"{view_node.outputs[0].name}_axes",
            values=np.array(sorted(removed), dtype=np.int64),
        )
        return new_perm, new_shape, [axes_const]
