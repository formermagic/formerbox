import logging
import os
from typing import Any, Dict, Iterable, NamedTuple, Optional, Text, cast

import jsonlines
from pydriller import Modification, RepositoryMining

logger = logging.getLogger(__name__)


class RepositoryDesc(NamedTuple):
    url: Text
    path: Text


def is_source_file(filename: Text) -> bool:
    return ".py" in filename


def parse_modification(modification: Modification) -> Optional[Dict[Text, Any]]:
    old_content = modification.source_code_before
    new_content = modification.source_code

    if not is_source_file(modification.filename):
        return None

    return {
        "old_content": old_content,
        "new_content": new_content,
        "old_path": modification.old_path,
        "new_path": modification.new_path,
        "diff": modification.diff,
    }


def parse_commits(repository: RepositoryDesc) -> Iterable[Dict[Text, Any]]:
    repository_mining = RepositoryMining(repository.path)
    logger.info("Started parsing commits for repository (%s)", repository.url)

    for commit in repository_mining.traverse_commits():
        if commit.merge:
            continue

        parsed_modifications = []
        for modification in commit.modifications:
            parsed_modification = parse_modification(modification)
            if parsed_modification is None:
                continue
            parsed_modifications.append(parsed_modification)

        if not parsed_modifications:
            logger.info(
                "Unable to parse modifications for commit %s, message: '%s'",
                commit.hash,
                commit.msg,
            )
            continue

        parsed_commit = {
            "commit_hash": commit.hash,
            "repository_name": commit.project_name,
            "repository_url": repository.url,
            "commit_message": commit.msg,
            "modifications": parsed_modifications,
        }

        yield parsed_commit

    logger.info("Finished parsing repository (%s)", repository.url)


def save_parsed_commits(
    output_path: Text, parsed_commits: Iterable[Dict[Text, Any]]
) -> None:
    if os.path.exists(output_path):
        os.remove(output_path)
    with jsonlines.open(output_path, mode="a") as writer:
        writer = cast(jsonlines.Writer, writer)
        for commit in parsed_commits:
            if commit is None:
                continue

            writer.write(commit)  # pylint: disable=no-member
