import logging
import re
from abc import abstractmethod

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.third_party.onnx_graphsurgeon import Constant

logger = logging.getLogger("onnxslim")


def get_name(name):
    """Sanitizes the input string by replacing illegal characters with underscores and prefixing with an underscore if
    numeric.
    """
    _illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")
    sanitized_name = _illegal_char_regex.sub("_", name)
    if sanitized_name.isdigit():
        sanitized_name = f"_{sanitized_name}"

    return sanitized_name


class NodeDescriptor:
    """
    case 0: input [1, 2, 3, 4, 5] output [0] Optype Name 5  1 i0 i1 i2 i3 i4 o0
    case 1: input [1, ...]        output [0] Optype Name 1+ 1 i0 o0
    case 2: input [..., 1, ...]   output [0] Optype Name 1* 1 i0 o0.
    """

    def __init__(self, node_spec):
        """Initialize NodeDescriptor with node_spec list requiring at least 4 elements."""
        if not isinstance(node_spec, list):
            raise ValueError("node_spec must be a list")
        if len(node_spec) < 4:
            raise ValueError(f"node_spec must have at least 4 elements {node_spec}")

        def get_input_info(io_spec):
            """Parses io_spec to return a tuple of (integer, boolean) indicating the presence of a plus sign in the
            input.
            """
            if not io_spec.isdigit():
                match = re.search(r"(\d+)([+*])", io_spec)
                if match:
                    number = match.group(1)
                    operator = match.group(2)

                    if not number.isdigit():
                        raise ValueError(f"input_num and output_num must be integers {io_spec}")

                    if operator == "+":
                        return int(number), True, "append"
                    elif operator == "*":
                        return int(number), True, "free-match"
                    else:
                        raise ValueError(f"operator must be + or * {io_spec}")

            return int(io_spec), False, None

        self.op = node_spec[0]
        self.name = node_spec[1]
        self.input_num, self.coarse_input_num, self.input_mode = get_input_info(node_spec[2])
        self.output_num, self.coarse_output_num, self.output_mode = get_input_info(node_spec[3])
        self.input_names = node_spec[4 : 4 + self.input_num]
        self.output_names = node_spec[4 + self.input_num :]
        assert len(self.input_names) == self.input_num
        assert len(self.output_names) == self.output_num, f"{self.name} {len(self.output_names)} != {self.output_num}"

    def __repr__(self):
        """Return a string representation of the object, including its name, operation type, input/output counts, and
        input/output names.
        """
        return f"name: {self.name}, type: {self.op}, input_num: {self.input_num}, output_num: {self.output_num}, input_names: {self.input_names}, output_names: {self.output_names}"

    def __dict__(self):
        """Returns a dictionary representation of the object, with 'name' as the key."""
        return {
            "name": self,
        }


class Pattern:
    def __init__(self, pattern):
        """Initialize the Pattern class with a given pattern and parse its nodes."""
        self.pattern = pattern
        self.nodes = self.parse_nodes()

    def parse_nodes(self):
        """Parse pattern into a list of NodeDescriptor objects from non-empty, stripped, and split lines."""
        nodes = self.pattern.split("\n")
        nodes = [line.strip().split() for line in nodes if line]
        nodes = [NodeDescriptor(node) for node in nodes if node]
        return nodes

    def match(self, node):
        """Match a node against a precompiled pattern."""
        return self.pattern.match(node)

    def __repr__(self):
        """Return a string representation of the pattern attribute."""
        return self.pattern


class PatternMatcher:
    def __init__(self, pattern, priority):
        """Initialize the PatternMatcher with a given pattern and priority, and prepare node references and output
        names.
        """
        self.pattern = pattern
        self.priority = priority
        self.pattern_dict = {node.name: node for node in pattern.nodes}
        self.output_names = [node.name for node in pattern.nodes if node.op == "output"]

    def get_match_point(self):
        """Retrieve the match point node from the pattern dictionary based on output node input names."""
        return self.pattern_dict[self.pattern_dict[self.output_names[0]].input_names[0]]

    def match(self, node):
        """Match a given node to a pattern by comparing input names with the match point node from the pattern
        dictionary.
        """
        match_point = self.get_match_point()

        def match_(node, pattern_node):
            """Match a given node to a pattern by comparing input names with the match point node from the pattern
            dictionary.
            """
            if pattern_node.op == "input":
                return True

            # node is an input variable
            if not hasattr(node, "op"):
                return False

            if node.op == pattern_node.op:
                setattr(self, pattern_node.name, node)

                node_feeds = node.feeds
                if pattern_node.coarse_input_num:
                    if len(node_feeds) < len(pattern_node.input_names):
                        return False
                else:
                    if len(node_feeds) != len(pattern_node.input_names):
                        return False

                if pattern_node.input_mode == "append" or pattern_node.input_mode is None:
                    pattern_nodes = [
                        self.pattern_dict[name] if name != "?" else None for name in pattern_node.input_names
                    ]
                    all_match = True
                    for node_feed, pattern_node in zip(node_feeds, pattern_nodes):
                        if pattern_node is not None:
                            node_match = match_(node_feed, pattern_node)
                            if not node_match:
                                return False
                            setattr(self, pattern_node.name, node_feed)

                    return all_match
                elif pattern_node.input_mode == "free-match":
                    pattern_nodes = [
                        self.pattern_dict[name] if name != "?" else None for name in pattern_node.input_names
                    ]
                    all_match = True
                    for pattern_node in pattern_nodes:
                        if pattern_node is not None:
                            node_match = False
                            for node_feed in node_feeds:
                                node_match = match_(node_feed, pattern_node)
                                if node_match:
                                    break
                            if not node_match:
                                return False
                            setattr(self, pattern_node.name, node_feed)

                    return all_match
            return False

        if match_(node, match_point):
            setattr(self, "output", node.outputs)
            if self.parameter_check():
                return True

        return False

    @abstractmethod
    def rewrite(self, opset=11):
        """Abstract method to rewrite the graph based on matched patterns, to be implemented by subclasses."""
        raise NotImplementedError("rewrite method must be implemented")

    def parameter_check(self):
        """Check and validate parameters, returning True if valid."""
        return True


class PatternGenerator:
    def __init__(self, onnx_model):
        """Initialize the PatternGenerator class with an ONNX model and process its graph."""
        self.graph = gs.import_onnx(onnx_model)
        self.graph.fold_constants().cleanup().toposort()

    def generate(self):
        """Generate the inputs, outputs, and nodes from the graph of the initialized ONNX model."""
        inputs = self.graph.inputs
        outputs = self.graph.outputs
        nodes = self.graph.nodes

        template = []
        for input in inputs:
            name = get_name(input.name)
            template.append(
                " ".join(
                    ["input", name, "0", str(len(input.outputs))] + [get_name(output.name) for output in input.outputs]
                )
            )

        for node in nodes:
            if node.op != "Constant":
                name = get_name(node.name)
                feeds = node.feeds
                users = node.users
                template.append(
                    " ".join(
                        [node.op, name, str(len(feeds)), str(len(users))]
                        + ["?" if isinstance(feed, Constant) else get_name(feed.name) for feed in feeds]
                        + ["?" if isinstance(user, Constant) else get_name(user.name) for user in users]
                    )
                )

        for output in outputs:
            name = get_name(output.name)
            template.append(
                " ".join(
                    ["output", name, str(len(output.inputs)), "0"] + [get_name(input.name) for input in output.inputs]
                )
            )

        return "\n".join(template)
