import os
import re
from typing import Any, Dict, List, Optional, Text

from commitizen import defaults
from commitizen.cz import BaseCommitizen
from commitizen.cz.utils import multiple_line_breaker, required_validator
from git import Commit


def parse_scope(text: Text) -> Text:
    if not text:
        return ""

    scope = text.strip().split()
    if len(scope) == 1:
        return scope[0]

    return "-".join(scope)


def parse_subject(text: Text) -> Text:
    if isinstance(text, str):
        text = text.strip(".").strip()

    return required_validator(text, msg="Subject is required.")


class GitneticCz(BaseCommitizen):
    bump_pattern = defaults.bump_pattern
    bump_map = defaults.bump_map
    commit_parser = (
        r"^(?P<change_type>feat|fix|refactor|perf|chore|build|BREAKING CHANGE)"
        r"(?:\((?P<scope>[^()\r\n]*)\)|\()?(?P<breaking>!)?:\s(?P<message>.*)?"
    )
    changelog_pattern = r"^(BREAKING[\-\ ]CHANGE|feat|fix|refactor|perf)(\(.+\))?(!)?"
    change_type_map = {
        "feat": "Features 222",
        "fix": "Fix",
        "refactor": "Refactor",
        "perf": "Performance",
        "chore": "Other",
        "build": "Build Process",
    }

    def questions(self) -> List[Dict[Text, Any]]:
        questions: List[Dict[Text, Any]] = [
            {
                "type": "list",
                "name": "prefix",
                "message": "Select the type of change you are committing",
                "choices": [
                    {
                        "value": "fix",
                        "name": "fix: A bug fix. Correlates with PATCH in SemVer",
                    },
                    {
                        "value": "feat",
                        "name": "feat: A new feature. Correlates with MINOR in SemVer",
                    },
                    {"value": "docs", "name": "docs: Documentation only changes"},
                    {
                        "value": "style",
                        "name": (
                            "style: Changes that do not affect the "
                            "meaning of the code (white-space, formatting,"
                            " missing semi-colons, etc)"
                        ),
                    },
                    {
                        "value": "refactor",
                        "name": (
                            "refactor: A code change that neither fixes "
                            "a bug nor adds a feature"
                        ),
                    },
                    {
                        "value": "perf",
                        "name": "perf: A code change that improves performance",
                    },
                    {
                        "value": "test",
                        "name": (
                            "test: Adding missing or correcting " "existing tests"
                        ),
                    },
                    {
                        "value": "build",
                        "name": (
                            "build: Changes that affect the build system or "
                            "external dependencies (example scopes: pip, docker, npm)"
                        ),
                    },
                    {
                        "value": "ci",
                        "name": (
                            "ci: Changes to our CI configuration files and "
                            "scripts (example scopes: GitLabCI)"
                        ),
                    },
                ],
            },
            {
                "type": "input",
                "name": "scope",
                "message": (
                    "Scope. Could be anything specifying place of the "
                    "commit change (users, db, poll):\n"
                ),
                "filter": parse_scope,
            },
            {
                "type": "input",
                "name": "subject",
                "filter": parse_subject,
                "message": (
                    "Subject. Concise description of the changes. "
                    "Imperative, lower case and no final dot:\n"
                ),
            },
            {
                "type": "confirm",
                "message": "Is this a BREAKING CHANGE? Correlates with MAJOR in SemVer",
                "name": "is_breaking_change",
                "default": False,
            },
            {
                "type": "input",
                "name": "body",
                "message": (
                    "Body. Motivation for the change and contrast this "
                    "with previous behavior:\n"
                ),
                "filter": multiple_line_breaker,
            },
            {
                "type": "input",
                "name": "footer",
                "message": (
                    "Footer. Information about Breaking Changes and "
                    "reference issues that this commit closes:\n"
                ),
            },
        ]
        return questions

    def message(self, answers: Dict[Text, Any]) -> Text:
        print(answers)
        prefix = answers["prefix"]
        scope = answers["scope"]
        subject = answers["subject"]
        body = answers["body"]
        footer = answers["footer"]
        is_breaking_change = answers["is_breaking_change"]

        if scope:
            scope = f"({scope})"
        if is_breaking_change:
            body = f"BREAKING CHANGE: {body}"
        if body:
            body = f"\n\n{body}"
        if footer:
            footer = f"\n\n{footer}"

        message = f"{prefix}{scope}: {subject}{body}{footer}"

        return message

    def example(self) -> Text:
        return (
            "fix: correct minor typos in code\n"
            "\n"
            "see the issue for details on the typos fixed\n"
            "\n"
            "closes issue #12"
        )

    def schema(self) -> Text:
        return (
            "<type>(<scope>): <subject>\n"
            "<BLANK LINE>\n"
            "(BREAKING CHANGE: )<body>\n"
            "<BLANK LINE>\n"
            "<footer>"
        )

    def schema_pattern(self) -> Text:
        pattern = (
            r"(build|ci|docs|feat|fix|perf|refactor|style|test|chore|revert|bump)"
            r"(\(\S+\))?:(\s.*)"
        )
        return pattern

    def info(self) -> Text:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filepath = os.path.join(dir_path, "conventional_commits_info.txt")
        with open(filepath, mode="r") as info_file:
            content = info_file.read()
        return content

    def process_commit(self, commit: str) -> str:
        pattern = re.compile(self.schema_pattern())
        message = re.match(pattern, commit)
        if message is None:
            return ""
        return message.group(3).strip()

    # pylint: disable=unused-argument
    def changelog_message_builder_hook(
        self, parsed_message: Dict[Text, Any], commit: Commit
    ) -> Dict[Text, Any]:
        return parsed_message

    # pylint: disable=unused-argument
    def changelog_hook(
        self, full_changelog: Text, partial_changelog: Optional[Text]
    ) -> Text:
        full_changelog = full_changelog.replace("##", "#")
        full_changelog = full_changelog.replace("###", "##")
        return full_changelog


discover_this = GitneticCz
