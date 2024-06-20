import re
from abc import ABCMeta, abstractmethod

from onnxslim.utils import logger
import onnxslim.onnx_graphsurgeon as gs
from onnxslim.onnx_graphsurgeon import Constant


def get_node_users(node):
    """Retrieve the list of nodes that use the outputs of the given node."""
    users = []
    for output in node.outputs:  # output is a Variable
        if len(output.outputs) == 0:
            users.append(output)
        users.extend(iter(output.outputs))
    return users


def get_node_feeds(node):
    """Retrieve the list of nodes that provide inputs to the given node."""
    feeds = []
    for input in node.inputs:
        if len(input.inputs) == 0 and not isinstance(input, Constant):
            feeds.append(input)
        elif isinstance(input, Constant):
            feeds.append(input)
        else:
            for feed in input.inputs:
                feeds.append(input if feed.op == "Split" else feed)
    return feeds


def get_name(name):
    _illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")
    sanitized_name = _illegal_char_regex.sub("_", name)
    if sanitized_name.isdigit():
        sanitized_name = f"_{sanitized_name}"

    return sanitized_name


class NodeDescriptor:
    def __init__(self, node_spec):
        if not isinstance(node_spec, list):
            raise ValueError('node_spec must be a list')
        if len(node_spec) < 4:
            raise ValueError(f'node_spec must have at least 4 elements {node_spec}')

        def get_input_info(io_spec):
            if not io_spec.isdigit():
                pattern_with_plus = re.search(r'(\d+)(\+)', io_spec)
                if pattern_with_plus:
                    return int(pattern_with_plus.group(1)), True
                else:
                    raise ValueError(f'input_num and output_num must be integers {io_spec}')

            return int(io_spec), False

        self.op = node_spec[0]
        self.name = node_spec[1]
        self.input_num, self.coarse_input_num = get_input_info(node_spec[2])
        self.output_num, self.coarse_output_num = get_input_info(node_spec[3])
        self.input_names = node_spec[4:4 + self.input_num]
        self.output_names = node_spec[4 + self.input_num:]
        assert len(self.input_names) == self.input_num
        assert len(self.output_names) == self.output_num, f'{self.name} {len(self.output_names)} != {self.output_num}'

    def __repr__(self):
        return f'name: {self.name}, type: {self.op}, input_num: {self.input_num}, output_num: {self.output_num}, input_names: {self.input_names}, output_names: {self.output_names}'

    def __dict__(self):
        return {
            'name': self,
        }


class Pattern:
    def __init__(self, pattern):
        self.pattern = pattern
        self.nodes = self.parse_nodes()

    def parse_nodes(self):
        nodes = self.pattern.split('\n')
        nodes = [line.strip().split() for line in nodes if line]
        nodes = [NodeDescriptor(node) for node in nodes if node]
        return nodes

    def match(self, node):
        return self.pattern.match(node)

    def __repr__(self):
        return self.pattern


class PatternMatcher:
    def __init__(self, pattern, priority):
        self.pattern = pattern
        self.priority = priority
        self.pattern_dict = {node.name: node for node in pattern.nodes}
        self.output_names = [node.name for node in pattern.nodes if node.op == 'output']

    def get_match_point(self):
        return self.pattern_dict[self.pattern_dict[self.output_names[0]].input_names[0]]

    def match(self, node):
        match_point = self.get_match_point()

        def match_(node, pattern_node):
            if pattern_node.op == 'input':
                return True

            # node is an input variable
            if not hasattr(node, 'op'):
                return False

            if node.op == pattern_node.op:
                setattr(self, pattern_node.name, node)

                node_feeds = get_node_feeds(node)
                if pattern_node.coarse_input_num:
                    if len(node_feeds) <= len(pattern_node.input_names):
                        return False
                else:
                    if len(node_feeds) != len(pattern_node.input_names):
                        logger.debug('len(node_feeds) != len(pattern_node.input_names)',
                                    len(node_feeds), len(pattern_node.input_names))
                        return False

                pattern_nodes = [self.pattern_dict[name] if name != "?" else None for name in pattern_node.input_names]
                all_match = True
                for node_feed, pattern_node in zip(node_feeds, pattern_nodes):
                    if pattern_node is not None:
                        node_match =  match_(node_feed, pattern_node)
                        if not node_match:
                            return False
                        setattr(self, pattern_node.name, node_feed)

                return all_match

            return False

        if match_(node, match_point):
            setattr(self, 'output', node.outputs)
            if self.parameter_check():
                return True

        return False

    @abstractmethod
    def rewrite(self):
        raise NotImplementedError('rewrite method must be implemented')

    def parameter_check(self):
        return True


class PatternGenerator:
    def __init__(self, onnx_model):
        self.graph = gs.import_onnx(onnx_model)
        self.graph.fold_constants().cleanup().toposort()

    def generate(self):
        inputs = self.graph.inputs
        outputs = self.graph.outputs
        nodes = self.graph.nodes

        template = []
        for input in inputs:
            name = get_name(input.name)
            template.append(' '.join(['input', name, '0', str(len(input.outputs))] +
                                     [get_name(output.name) for output in input.outputs]))

        for node in nodes:
            if node.op != "Constant":
                name = get_name(node.name)
                feeds = get_node_feeds(node)
                users = get_node_users(node)
                template.append(' '.join([node.op, name, str(len(feeds)), str(len(users))] +
                                         [get_name(feed.name) if not isinstance(feed, Constant) else '?' for feed in feeds] +
                                         [get_name(user.name) if not isinstance(user, Constant) else '?' for user in users]))

        for output in outputs:
            name = get_name(output.name)
            template.append(' '.join(['output', name, str(len(output.inputs)), '0'] + [get_name(input.name) for input in output.inputs]))

        return '\n'.join(template)
