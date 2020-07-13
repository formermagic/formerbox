import os
from typing import Any, Dict, Iterable, Optional, Text, cast

import jsonlines
from pydriller import Modification, RepositoryMining


class RepositoryDesc:
    def __init__(self, repo_url: Text, repo_path: Text) -> None:
        self.repo_url = repo_url
        self.repo_path = repo_path


class GitCommitParser:
    def __init__(self, repository: RepositoryDesc) -> None:
        self.repository = repository

    @staticmethod
    def _is_source_file(filename: Text) -> bool:
        return ".py" in filename

    def _parse_modification(
        self, modification: Modification
    ) -> Optional[Dict[Text, Any]]:
        old_content = modification.source_code_before
        new_content = modification.source_code

        if not self._is_source_file(modification.filename):
            return None

        return {
            "old_content": old_content,
            "new_content": new_content,
            "old_path": modification.old_path,
            "new_path": modification.new_path,
            "diff": modification.diff,
        }

    def _parse_commits(self) -> Iterable[Dict[Text, Any]]:
        repository_path = self.repository.repo_path
        repository_mining = RepositoryMining(repository_path)
        for commit in repository_mining.traverse_commits():
            if commit.merge:
                continue

            parsed_modifications = []
            for modification in commit.modifications:
                parsed_modification = self._parse_modification(modification)
                if parsed_modification is None:
                    continue
                parsed_modifications.append(parsed_modification)

            if not parsed_modifications:
                continue

            parsed_commit = {
                "commit_hash": commit.hash,
                "repository_name": commit.project_name,
                "repository_url": self.repository.repo_url,
                "commit_message": commit.msg,
                "modifications": parsed_modifications,
            }

            yield parsed_commit

    @staticmethod
    def _save_parsed_commits(
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

    def parse(self, output_path: Text) -> None:
        parsed_commits = self._parse_commits()
        self._save_parsed_commits(output_path, parsed_commits)
