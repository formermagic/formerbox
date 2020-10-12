import math
import shutil
import tempfile
from collections import deque
from typing import Iterator, List, Optional, Text, Tuple

from tree_sitter import Language, Node, Parser

from .git_repo_extractor import clone_repository

PYTHON_LANG_REPO_URL = "https://github.com/tree-sitter/tree-sitter-python"


class TreeNode:
    def __init__(
        self,
        node: Node,
        start_point: Optional[Tuple[int, int]] = None,
        end_point: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.node: Node = node
        self.children: List[Node] = node.children
        self.__start_point = start_point or node.start_point
        self.__end_point = end_point or node.end_point

    def sexp(self) -> Text:
        return self.node.sexp()

    @property
    def start_point(self) -> Tuple[int, int]:
        return self.__start_point

    @property
    def end_point(self) -> Tuple[int, int]:
        return self.__end_point

    @property
    def type(self) -> Text:
        return self.node.type

    @property
    def is_named(self) -> bool:
        return self.node.is_named


class LanguageRepr:
    def __init__(self, library_path: Text, lang: Text) -> None:
        self.__library_path = library_path
        self.__lang = lang
        self.language = self.__built_language()

    def __built_language(self) -> Language:
        return Language(self.__library_path, self.__lang)

    @property
    def parser(self) -> Parser:
        _parser = Parser()
        _parser.set_language(self.language)
        return _parser


class LanguageReprBuilder:
    @staticmethod
    def build(
        url: Text, lang: Text, output_path: Text, removing_tmp: bool = True
    ) -> LanguageRepr:
        # clone a git repository to a temp dir
        tmp_dirname = tempfile.mkdtemp()
        repository_path = clone_repository(url, tmp_dirname)
        # build a language library with the given grammar
        Language.build_library(output_path, repo_paths=[repository_path])
        # remove temp dir if needed
        if removing_tmp:
            shutil.rmtree(repository_path)
        # construct a language repr wrapper
        lang_repr = LanguageRepr(library_path=output_path, lang=lang)
        return lang_repr


class ASTParser:
    def __init__(self, language_repr: LanguageRepr) -> None:
        self.__language_repr = language_repr
        self.parser = language_repr.parser

    def parse_all_nodes(
        self, program: Text, parse_def: bool = True
    ) -> List[Tuple[Text, Text]]:
        tree = self.parser.parse(bytes(program, "utf8"))
        root_node = TreeNode(tree.root_node)

        program_lines = program.split("\n")
        src_lines: List[Text] = []
        ast_lines: List[Text] = []

        for node in self.traverse_tree(root_node):
            if parse_def:
                parsed_def: Optional[Tuple[Text, Text]] = None
                if node.type in ["class_definition", "function_definition"]:
                    parsed_def = self.__parse_def(node, program_lines)
                elif node.type == "decorated_definition":
                    parsed_def = self.__parse_decorated_def(node, program_lines)

                if parsed_def is not None:
                    src, ast = parsed_def
                    src_lines.append(src)
                    ast_lines.append(ast)

            node_types = ["definition", "statement", "primary_expression", "comment"]
            if self.contains_one_of(node.type, node_types):
                src = self.find_substring(program_lines, node)
                ast = self.parse_node(node, program_lines)
                if src and ast:
                    src_lines.append(src)
                    ast_lines.append(ast)

        return list(zip(src_lines, ast_lines))

    def parse_root_nodes(
        self, program: Text, parse_def: bool = True, parse_class_nodes: bool = True
    ) -> List[Tuple[Text, Text]]:
        tree = self.parser.parse(bytes(program, "utf8"))
        program_lines = program.split("\n")
        raw_bodies: List[Text] = []
        parsed_children: List[Text] = []

        children = deque([TreeNode(node) for node in tree.root_node.children])

        while children:
            root_child = children.popleft()

            if parse_def:
                parsed_def: Optional[Tuple[Text, Text]] = None
                if root_child.type == "class_definition":
                    # schedule class children nodes parsing
                    if parse_class_nodes:
                        block_node = root_child.children[-1]
                        for node in block_node.children[::-1]:
                            children.appendleft(TreeNode(node))

                    parsed_def = self.__parse_def(root_child, program_lines)
                elif root_child.type == "function_definition":
                    parsed_def = self.__parse_def(root_child, program_lines)
                elif root_child.type == "decorated_definition":
                    # parse wrapped node into a decorator as well
                    for node in root_child.children[::-1]:
                        children.appendleft(TreeNode(node))

                if parsed_def is not None:
                    source_code, parsed_ast = parsed_def
                    raw_bodies.append(source_code)
                    parsed_children.append(parsed_ast)

            parsed_node = self.parse_node(root_child, program_lines)
            if parsed_node is not None:
                raw_bodies.append(self.find_substring(program_lines, root_child))
                parsed_children.append(parsed_node)

        return list(zip(raw_bodies, parsed_children))

    @staticmethod
    def contains_one_of(text: Text, substrings: List[Text]) -> bool:
        return any(substr in text for substr in substrings)

    def traverse_tree(self, node: TreeNode) -> Iterator[TreeNode]:
        node_deque = deque(node.children)
        while node_deque:
            left_node = TreeNode(node_deque.popleft())
            if left_node.is_named:
                node_deque.extendleft(left_node.children[::-1])
                yield left_node

    def __parse_def(
        self, node: TreeNode, program_lines: List[Text]
    ) -> Optional[Tuple[Text, Text]]:
        # find where the definition ends
        start_point = node.children[0].start_point
        end_point = node.children[0].end_point
        for child in node.children:
            if child.type == "block":
                break
            end_point = child.end_point

        definition_node = TreeNode(node.node, start_point, end_point)
        source_code = self.find_substring(program_lines, definition_node)
        parsed_ast = self.parse_node(definition_node, program_lines)

        if source_code and parsed_ast:
            # drop body from node definition
            parsed_ast = parsed_ast.split(" body:")[0] + ")"
            return source_code, parsed_ast

        return None

    def __parse_decorated_def(
        self, node: TreeNode, program_lines: List[Text]
    ) -> Optional[Tuple[Text, Text]]:
        # find where the definition ends
        start_point = node.children[0].start_point
        def_nodes = node.children[-1].children
        end_point = def_nodes[0].end_point
        for child in def_nodes:
            if child.type == "block":
                break
            end_point = child.end_point

        definition_node = TreeNode(node.node, start_point, end_point)
        src = self.find_substring(program_lines, definition_node)
        ast = self.parse_node(definition_node, program_lines)

        if src and ast:
            # drop body from node definition
            ast = ast.split(" body:")[0] + ")" + ")"
            return src, ast

        return None

    def parse_node(self, node: TreeNode, program_lines: List[Text]) -> Optional[Text]:
        identifiers = []
        for child_node in self.traverse_tree(node):
            if child_node.type == "identifier":
                identifier = self.find_substring(program_lines, child_node)
                identifiers.append(identifier)

            # :workaround: tree_sitter escape sequences string buggy parsing
            elif child_node.type == "string":
                distance = self.distance(child_node.start_point, child_node.end_point)
                if distance > 100:
                    return None

        source_sexp = node.sexp()
        for identifier in identifiers:
            source_sexp = source_sexp.replace("identifier", identifier, 1)

        return source_sexp

    def find_substring(self, program_lines: List[Text], node: TreeNode) -> Text:
        start_point, end_point = node.start_point, node.end_point
        lines: List[Text] = []

        for idx in range(start_point[0], end_point[0] + 1):
            start_idx, end_idx = 0, len(program_lines[idx])
            if idx == start_point[0]:
                start_idx = start_point[1]
            if idx == end_point[0]:
                end_idx = end_point[1]
            lines.append(program_lines[idx][start_idx:end_idx])

        return "\n".join(lines)

    @staticmethod
    def distance(lhs: Tuple[int, int], rhs: Tuple[int, int]) -> float:
        """Calculate euclidean distance between 2-dimensional points."""
        return math.sqrt(math.pow(rhs[0] - lhs[0], 2) + math.pow(rhs[1] - lhs[1], 2))
