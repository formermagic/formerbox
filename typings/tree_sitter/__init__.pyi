from typing import Any, List, Optional, Tuple, Text

TSFieldId = int

class TSTree: ...
class TSTreeCursor: ...
class TSNode: ...
class TSParser: ...
class TSQuery: ...

class Language:
    @staticmethod
    def build_library(output_path: Text, repo_paths: List[Text]) -> bool: ...
    def __init__(self, library_path: Text, name: Text) -> None: ...
    def field_id_for_name(self, name: Text) -> TSFieldId: ...
    def query(self, source: Text) -> "Query": ...

class Query:
    query: TSQuery
    capture_names: List[Text]

    # :methods:
    def matches(self, node: "Node") -> None: ...
    def captures(self, node: "Node") -> List[Tuple["Node", Text]]: ...

class Parser:
    parser: TSParser

    # :methods:
    def parse(
        self, bytes: bytes, old_tree: Optional[Tree] = None
    ) -> Optional[Tree]: ...
    def set_language(self, language: Language) -> None: ...

class Tree:
    tree: TSTree

    # :accessors:
    @property
    def root_node(self) -> "Node": ...
    # :methods:
    def walk(self) -> "TreeCursor": ...
    def edit(
        self,
        start_byte: int,
        old_end_byte: int,
        new_end_byte: int,
        start_point: Tuple[int, int],
        old_end_point: Tuple[int, int],
        new_end_point: Tuple[int, int],
    ) -> None: ...

class TreeCursor:
    cursor: TSTreeCursor
    node: "Node"
    tree: Tree

    # :methods:
    def current_field_name(self) -> Text: ...
    def goto_parent(self) -> bool: ...
    def goto_first_child(self) -> bool: ...
    def goto_next_sibling(self) -> bool: ...

class Node:
    node: TSNode
    children: List["Node"]
    tree: Tree

    start_byte: int
    start_point: Tuple[int, int]
    end_byte: int
    end_point: Tuple[int, int]

    has_changes: bool
    has_error: bool
    is_named: bool

    type: Text

    # :methods:
    def walk(self) -> TreeCursor: ...
    def child_by_field_id(self, id: TSFieldId) -> "Node": ...
    def child_by_field_name(self, name: Text) -> None: ...
    def sexp(self) -> Text: ...

