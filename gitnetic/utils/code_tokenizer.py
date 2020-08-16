# Copyright (c) 2019-present, Facebook, Inc.
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
    "#": " STOKEN0 ",
    "\\n": " STOKEN1 ",
    '"""': " STOKEN2 ",
    "'''": " STOKEN3 ",
}

PYTHON_TOKEN2CHAR = {
    "STOKEN0": "#",
    "STOKEN1": "\\n",
    "STOKEN2": '"""',
    "STOKEN3": "'''",
}

tokenize_v14_international = TokenizerV14International()


class SpecialToken(Enum):
    DEDENT = "DEDENT"
    ENCDOM = "ENDCOM"
    ENDMARKER = "ENDMARKER"
    INDENT = "INDENT"
    NEW_LINE = "NEW_LINE"
    SPACE = "▁"
    SPACETOKEN = "SPACETOKEN"
    STRNEWLINE = "STRNEWLINE"
    TABSYMBOL = "TABSYMBOL"


def process_string(
    token: Text,
    char2token: Dict[Text, Text],
    token2char: Dict[Text, Text],
    is_comment: bool,
) -> Text:
    if is_comment:
        token = re.sub(" +", " ", token)
        token = re.sub(r"(.)\1\1\1\1+", r"\1\1\1\1\1", token)
        if len(re.sub(r"\W", "", token)) < 2:
            return ""

    token = token.replace(" ", " ▁ ")
    for char, special_token in char2token.items():
        token = token.replace(char, special_token)

    if token.startswith(" STOKEN0"):
        if token.endswith("\n"):
            token = token[:-1]
        token += f" {SpecialToken.ENCDOM.value}"

    token = token.replace("\n", f" {SpecialToken.STRNEWLINE.value} ")
    token = token.replace("\t", f" {SpecialToken.TABSYMBOL.value} ")
    token = token.replace("\r", "")
    token = re.sub(" +", " ", token)
    token = tokenize_v14_international(token)
    token = re.sub(" +", " ", token)

    for special_token, char in token2char.items():
        token = token.replace(special_token, char)

    return token


# pylint: disable=no-else-continue, disable=broad-except
def tokenize_python(text: Text, keep_comments: bool = False) -> List[Text]:
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
            ):
                raise Exception(
                    f'Impossible to parse tokens because icorrect source code "{text[0:30]}" ...'
                )
            except StopIteration:
                raise Exception("End of iterator before ENDMARKER token.")

            if token_info.type in [tokenize.ENCODING, tokenize.NL]:
                continue

            elif token_info.type == tokenize.NEWLINE:
                if removed_docstr == 1:
                    removed_docstr = 0
                    continue
                tokens.append(SpecialToken.NEW_LINE.value)

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
                tokens.append(SpecialToken.INDENT.value)

            elif token_info.type == tokenize.DEDENT:
                # empty block
                if tokens[-1] == SpecialToken.INDENT.value:
                    tokens = tokens[:-1]
                else:
                    tokens.append(SpecialToken.DEDENT.value)

            elif token_info.type == tokenize.ENDMARKER:
                tokens.append(SpecialToken.ENDMARKER.value)
                break

            else:
                tokens.append(token_info.string)

        assert tokens[-1] == SpecialToken.ENDMARKER.value, "Error, no end marker"
        return tokens[:-1]
    except KeyboardInterrupt as err:
        raise err
    except BaseException:
        return []


# pylint: disable=no-else-continue, disable=broad-except
def detokenize_python(tokens: Union[List[Text], Text]) -> Text:
    try:
        assert isinstance(tokens, (str, list))
        if isinstance(tokens, list):
            tokens = " ".join(tokens)
        tokens = tokens.replace(SpecialToken.ENCDOM.value, SpecialToken.NEW_LINE.value)
        tokens = tokens.replace(SpecialToken.SPACE.value, SpecialToken.SPACETOKEN.value)

        lines = tokens.split(SpecialToken.NEW_LINE.value)
        tabs = ""
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith(f"{SpecialToken.INDENT.value} "):
                tabs += "    "
                line = line.replace(f"{SpecialToken.INDENT.value} ", tabs)
            elif line.startswith(SpecialToken.DEDENT.value):
                number_dedent = line.count(SpecialToken.DEDENT.value)
                tabs = tabs[4 * number_dedent :]
                line = line.replace(SpecialToken.DEDENT.value, "")
                line = line.strip()
                line = tabs + line
            elif line == SpecialToken.DEDENT.value:
                line = ""
            else:
                line = tabs + line
            lines[i] = line

        untokenized = "\n".join(lines)

        # find string and comment with parser and detokenize string correctly
        try:
            for token_info in tokenize.tokenize(
                BytesIO(untokenized.encode("utf-8")).readline
            ):
                if token_info.type in [tokenize.STRING, tokenize.COMMENT]:
                    token = (
                        token_info.string.replace(SpecialToken.STRNEWLINE.value, "\n")
                        .replace(SpecialToken.TABSYMBOL.value, "\t")
                        .replace(" ", "")
                        .replace(SpecialToken.SPACETOKEN.value, " ")
                    )
                    untokenized = untokenized.replace(token_info.string, token)
        except KeyboardInterrupt as err:
            raise err
        except BaseException:
            pass

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

