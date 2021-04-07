# Copyright (c) 2019-present, Facebook, Inc. and The FormerMagic Inc. team.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import tokenize
from enum import Enum
from io import BytesIO
from typing import Dict, List, Text, Union

from sacrebleu.tokenizers import TokenizerV14International

PYTHON_CHAR2TOKEN = {
    "#": "<STOKEN0>",
    "\\n": "<STOKEN1>",
    '"""': "<STOKEN2>",
    "'''": "<STOKEN3>",
}

PYTHON_TOKEN2CHAR = {
    "<STOKEN0>": "#",
    "<STOKEN1>": "\\n",
    "<STOKEN2>": '"""',
    "<STOKEN3>": "'''",
}

tokenize_v14_international = TokenizerV14International()


class SpecialToken(Enum):
    dedent_token = "<dedent>"
    encdom_token = "<encdom>"
    endmarker_token = "<endmarker>"
    indent_token = "<indent>"
    newline_token = "<newline>"
    space_token = "▁"
    spacetoken_token = "<spacetoken>"
    strnewline_token = "<strnewline>"
    tabsymbol_token = "<tabsymbol>"


SPECIAL_TOKENS = set(token.value for token in SpecialToken)


def __stringify(tokens: List[str], add_token_space: bool) -> str:
    """Joins tokens into line with or without spaces aroung special tokens."""
    line = " ".join(tokens)
    if not add_token_space:
        tokens_pattern = "|".join(SPECIAL_TOKENS)
        line = re.sub(r"\s*(%s)\s*" % tokens_pattern, r"\1", line)
    return line


def __destringify(line: str, add_token_space: bool) -> str:
    """Adds spaces to special tokens in the tokenized line if they are missing."""
    if not add_token_space:
        tokens_pattern = "|".join(SPECIAL_TOKENS)
        line = re.sub(r"\s*(%s)\s*" % tokens_pattern, r" \1 ", line)
    return line


def process_string(
    token: Text,
    char2token: Dict[Text, Text],
    token2char: Dict[Text, Text],
    is_comment: bool,
    use_bleu_tokenization: bool = False,
) -> Text:
    if is_comment:
        token = re.sub(" +", " ", token)
        token = re.sub(r"(.)\1\1\1\1+", r"\1\1\1\1\1", token)
        if len(re.sub(r"\W", "", token)) < 2:
            return ""

    token = token.replace(" ", f" {SpecialToken.space_token.value} ")
    for char, special_token in char2token.items():
        token = token.replace(char, special_token)

    if token.startswith("<STOKEN0>"):
        if token.endswith("\n"):
            token = token[:-1]
        token += f" {SpecialToken.encdom_token.value}"

    token = token.replace("\n", f" {SpecialToken.strnewline_token.value} ")
    token = token.replace("\t", f" {SpecialToken.tabsymbol_token.value} ")
    token = re.sub(" +", " ", token)
    if use_bleu_tokenization:
        token = tokenize_v14_international(token)
    token = re.sub(" +", " ", token)

    # split string prefix if one stands at the beginning
    regex = r"""([bruf]*) ((\"""|'''|"|')(?:(?!\3)(?:\\.|[^\\]))*\3)"""
    token = re.sub(regex, r"\1\2", token)

    for special_token, char in token2char.items():
        token = token.replace(special_token, char)

    token = token.replace("\r", "")

    return token


# pylint: disable=no-else-continue, disable=broad-except
def tokenize_python(
    text: Text,
    keep_comments: bool = False,
    add_token_space: bool = False,
) -> Text:
    try:
        assert isinstance(text, str)
        text = text.replace(r"\r", "")
        tokens = []

        try:
            text_bytes = BytesIO(text.encode("utf-8"))
            iterator = tokenize.tokenize(text_bytes.readline)
        except SyntaxError as err:
            raise err

        removed_docstr = 0
        while True:
            try:
                token_info = next(iterator)
            except (
                tokenize.TokenError,
                IndentationError,
                SyntaxError,
                UnicodeDecodeError,
            ) as err:
                raise Exception(
                    f'Impossible to parse tokens because icorrect source code "{text[0:30]}" ...'
                ) from err
            except StopIteration as err:
                raise Exception("End of iterator before ENDMARKER token.") from err

            if token_info.type in [tokenize.ENCODING, tokenize.NL]:
                continue

            elif token_info.type == tokenize.NEWLINE:
                if removed_docstr == 1:
                    removed_docstr = 0
                    continue
                tokens.append(SpecialToken.newline_token.value)

            elif token_info.type == tokenize.COMMENT:
                if keep_comments:
                    com = process_string(
                        token_info.string,
                        PYTHON_CHAR2TOKEN,
                        PYTHON_TOKEN2CHAR,
                        is_comment=True,
                    )
                    if len(com) > 0:
                        tokens.append(com)
                else:
                    continue

            elif token_info.type == tokenize.STRING:
                if token_info.string == token_info.line.strip():  # docstring
                    if not keep_comments:
                        removed_docstr = 1
                        continue
                    else:
                        coms = process_string(
                            token_info.string,
                            PYTHON_CHAR2TOKEN,
                            PYTHON_TOKEN2CHAR,
                            is_comment=True,
                        )
                        if len(coms) > 0:
                            tokens.append(coms)
                        else:
                            removed_docstr = 1
                else:
                    tokens.append(
                        process_string(
                            token_info.string,
                            PYTHON_CHAR2TOKEN,
                            PYTHON_TOKEN2CHAR,
                            is_comment=False,
                        )
                    )

            elif token_info.type == tokenize.INDENT:
                tokens.append(SpecialToken.indent_token.value)

            elif token_info.type == tokenize.DEDENT:
                # empty block
                if tokens[-1] == SpecialToken.indent_token.value:
                    tokens = tokens[:-1]
                else:
                    tokens.append(SpecialToken.dedent_token.value)

            elif token_info.type == tokenize.ENDMARKER:
                tokens.append(SpecialToken.endmarker_token.value)
                break

            else:
                tokens.append(token_info.string)

        assert tokens[-1] == SpecialToken.endmarker_token.value, "Error, no end marker"

    except KeyboardInterrupt as err:
        raise err
    except BaseException:
        tokens = []

    return __stringify(tokens[:-1], add_token_space)


# pylint: disable=no-else-continue, disable=broad-except
def detokenize_python(
    tokens: Union[List[Text], Text],
    add_token_space: bool = False,
) -> Text:
    try:
        assert isinstance(tokens, (str, list))
        if isinstance(tokens, list):
            tokens = " ".join(tokens)

        # restore missing spaces around special tokens
        tokens = __destringify(tokens, add_token_space)

        # prepare tokenized string for splitting
        tokens = (
            tokens.replace(
                SpecialToken.encdom_token.value,
                SpecialToken.newline_token.value,
            )
            .replace(
                SpecialToken.space_token.value,
                SpecialToken.spacetoken_token.value,
            )
            .replace(
                SpecialToken.strnewline_token.value,
                SpecialToken.newline_token.value,
            )
        )

        # prepare tokenized lines for detokenization
        lines = tokens.split(SpecialToken.newline_token.value)

        tabs = ""
        for idx, line in enumerate(lines):
            line = line.strip()
            if line.startswith(f"{SpecialToken.indent_token.value} "):
                tabs += "    "
                line = line.replace(f"{SpecialToken.indent_token.value} ", tabs)
            elif line.startswith(SpecialToken.dedent_token.value):
                number_dedent = line.count(SpecialToken.dedent_token.value)
                tabs = tabs[4 * number_dedent :]
                line = line.replace(SpecialToken.dedent_token.value, "")
                line = line.strip()
                line = tabs + line
            elif line == SpecialToken.dedent_token.value:
                line = ""
            else:
                line = tabs + line

            # python tokenizer ignores comment indentation
            # this way we check if comment should get more tabs
            try:
                next_line = lines[idx + 1].strip()
                if line.strip().startswith("#"):
                    if next_line.startswith(f"{SpecialToken.indent_token.value} "):
                        line = "    " + line
            except IndexError:
                pass

            lines[idx] = line

        untokenized = "\n".join(lines)

        # detokenize string and comment
        untokenized = (
            untokenized.replace(SpecialToken.strnewline_token.value, "\n")
            .replace(SpecialToken.tabsymbol_token.value, "\t")
            .replace(f" {SpecialToken.spacetoken_token.value} ", " ")
            .replace(SpecialToken.spacetoken_token.value, " ")
        )

        # detokenize imports
        untokenized = (
            untokenized.replace(". ", ".")
            .replace(" .", ".")
            .replace("import.", "import .")
            .replace("from.", "from .")
        )

        untokenized = untokenized.replace("> >", ">>").replace("< <", "<<")
        return untokenized
    except KeyboardInterrupt as err:
        raise err
    except BaseException:
        return ""
