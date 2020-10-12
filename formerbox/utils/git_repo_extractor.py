import logging
import os
from typing import Text

from git import Repo

logger = logging.getLogger(__name__)


def convert_url_to_name(url: Text) -> Text:
    name = os.path.basename(url)
    name = os.path.splitext(name)[0]
    return name


def clone_repository(url: Text, output_path: Text) -> Text:
    logger.info("Started cloning repo with url: %s", url)

    # extract basename from the given url to clone
    basename = convert_url_to_name(url)
    # compile a path to clone repository to
    destpath = os.path.join(output_path, basename)
    # clone repository to the built destination path
    Repo.clone_from(url=url, to_path=destpath)

    logging.info("Cloned %s into %s", basename, destpath)

    return destpath
