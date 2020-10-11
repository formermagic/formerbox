# coding=utf-8
# Copyright 2020 AllenNLP Authors and The FormerMagic Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
import pkgutil
import sys
from contextlib import contextmanager
from importlib.machinery import FileFinder
from pathlib import Path
from typing import Generator, Text, TypeVar, Union

T = TypeVar("T")  # pylint: disable=invalid-name
PathType = Union[os.PathLike, Text]
ContextManagerFunctionReturnType = Generator[T, None, None]


@contextmanager
def push_python_path(path: PathType) -> ContextManagerFunctionReturnType[None]:
    """
    Prepends the given path to `sys.path`.
    This method is intended to use with `with`, so after its usage, its value willbe removed from
    `sys.path`.
    """
    # In some environments, such as TC, it fails when sys.path
    # contains a relative path, such as ".".
    path = Path(path).resolve()
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        # Better to remove by value, in case `sys.path` was manipulated in between.
        sys.path.remove(path)


def import_module_and_submodules(package_name: Text) -> None:
    """
    Import all submodules under the given package.
    Primarily useful so that people using AllenNLP as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    """
    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    with push_python_path("."):
        # Import at top level
        module = importlib.import_module(package_name)
        path = getattr(module, "__path__", [])
        path_string = "" if not path else path[0]

        # walk_packages only finds immediate children, so need to recurse.
        for module_finder, name, _ in pkgutil.walk_packages(path):
            # Sometimes when you import third-party libraries that are on your path,
            # `pkgutil.walk_packages` returns those too, so we need to skip them.
            if isinstance(module_finder, FileFinder):
                if path_string and module_finder.path != path_string:
                    continue

            subpackage = f"{package_name}.{name}"
            import_module_and_submodules(subpackage)


def import_user_module(user_dir: Text) -> Text:
    # prepare the module path
    module_path = os.path.abspath(user_dir)
    if not os.path.exists(module_path):
        rel_path = os.path.join(os.path.dirname(__file__), "..", user_dir)
        if os.path.exists(rel_path):
            module_path = rel_path

    # get the module_path components
    module_parent, module_name = os.path.split(module_path)

    # check if module is imported and cached
    if module_name in sys.modules:
        module_obj = sys.modules[module_name]
        del sys.modules[module_name]
    else:
        module_obj = None

    # import the module
    sys.path.insert(0, module_parent)
    importlib.import_module(module_name)

    # cache the imported module
    sys.modules["plugin_user_dir"] = sys.modules[module_name]
    if module_obj is not None and module_name != "plugin_user_dir":
        sys.modules[module_name] = module_obj

    return module_name
