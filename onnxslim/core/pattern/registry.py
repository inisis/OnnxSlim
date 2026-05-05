from __future__ import annotations

from collections import OrderedDict

# Maps fusion name -> list of matchers. Multiple matchers may share a name when
# an op's semantics change across opsets; the dispatcher picks the one whose
# `applies_to(graph.opset)` is true. Order within the list is registration order.
DEFAULT_FUSION_PATTERNS: "OrderedDict[str, list]" = OrderedDict()


def register_fusion_pattern(fusion_pattern=None, *, priority=None, min_opset=None, max_opset=None):
    """Register a fusion pattern. Supports two calling styles:

    Direct (legacy):
        register_fusion_pattern(MyMatcher(1))

    Class decorator (preferred for opset-aware matchers):
        @register_fusion_pattern(priority=1, max_opset=12)
        class MyMatcher(PatternMatcher): ...

    Multiple matchers may share a name when an op's semantics change across opsets;
    `min_opset` / `max_opset` (inclusive) bracket which graph opsets the matcher applies to.
    """
    if fusion_pattern is not None:
        DEFAULT_FUSION_PATTERNS.setdefault(fusion_pattern.name, []).append(fusion_pattern)
        return fusion_pattern

    def decorator(cls):
        instance = cls(priority)
        if min_opset is not None:
            instance.min_opset = min_opset
        if max_opset is not None:
            instance.max_opset = max_opset
        DEFAULT_FUSION_PATTERNS.setdefault(instance.name, []).append(instance)
        return cls

    return decorator


def get_fusion_patterns(skip_fusion_patterns: str | None = None):
    """Returns a copy of the default fusion patterns, optionally excluding patterns by name."""
    default_fusion_patterns = OrderedDict((k, list(v)) for k, v in DEFAULT_FUSION_PATTERNS.items())
    if skip_fusion_patterns:
        for pattern in skip_fusion_patterns:
            default_fusion_patterns.pop(pattern, None)

    return default_fusion_patterns


from .elimination import *
from .fusion import *
