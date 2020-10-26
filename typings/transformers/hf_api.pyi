"""
This type stub file was generated by pyright.
"""

import io
from typing import Dict, List, Optional, Tuple

ENDPOINT = "https://huggingface.co"
class S3Obj:
    """
    Data structure that represents a file belonging to the current user.
    """
    def __init__(self, filename: str, LastModified: str, ETag: str, Size: int, **kwargs) -> None:
        ...
    


class PresignedUrl:
    def __init__(self, write: str, access: str, type: str, **kwargs) -> None:
        ...
    


class S3Object:
    """
    Data structure that represents a public file accessible on our S3.
    """
    def __init__(self, key: str, etag: str, lastModified: str, size: int, rfilename: str, **kwargs) -> None:
        ...
    


class ModelInfo:
    """
    Info about a public model accessible from our S3.
    """
    def __init__(self, modelId: str, key: str, author: Optional[str] = ..., downloads: Optional[int] = ..., tags: List[str] = ..., pipeline_tag: Optional[str] = ..., siblings: Optional[List[Dict]] = ..., **kwargs) -> None:
        ...
    


class HfApi:
    def __init__(self, endpoint=...) -> None:
        ...
    
    def login(self, username: str, password: str) -> str:
        """
        Call HF API to sign in a user and get a token if credentials are valid.

        Outputs:
            token if credentials are valid

        Throws:
            requests.exceptions.HTTPError if credentials are invalid
        """
        ...
    
    def whoami(self, token: str) -> Tuple[str, List[str]]:
        """
        Call HF API to know "whoami"
        """
        ...
    
    def logout(self, token: str) -> None:
        """
        Call HF API to log out.
        """
        ...
    
    def presign(self, token: str, filename: str, organization: Optional[str] = ...) -> PresignedUrl:
        """
        Call HF API to get a presigned url to upload `filename` to S3.
        """
        ...
    
    def presign_and_upload(self, token: str, filename: str, filepath: str, organization: Optional[str] = ...) -> str:
        """
        Get a presigned url, then upload file to S3.

        Outputs:
            url: Read-only url for the stored file on S3.
        """
        ...
    
    def list_objs(self, token: str, organization: Optional[str] = ...) -> List[S3Obj]:
        """
        Call HF API to list all stored files for user (or one of their organizations).
        """
        ...
    
    def delete_obj(self, token: str, filename: str, organization: Optional[str] = ...):
        """
        Call HF API to delete a file stored by user
        """
        ...
    
    def model_list(self) -> List[ModelInfo]:
        """
        Get the public list of all the models on huggingface, including the community models
        """
        ...
    


class TqdmProgressFileReader:
    """
    Wrap an io.BufferedReader `f` (such as the output of `open(…, "rb")`)
    and override `f.read()` so as to display a tqdm progress bar.

    see github.com/huggingface/transformers/pull/2078#discussion_r354739608
    for implementation details.
    """
    def __init__(self, f: io.BufferedReader) -> None:
        ...
    
    def close(self):
        ...
    


class HfFolder:
    path_token = ...
    @classmethod
    def save_token(cls, token):
        """
        Save token, creating folder as needed.
        """
        ...
    
    @classmethod
    def get_token(cls):
        """
        Get token or None if not existent.
        """
        ...
    
    @classmethod
    def delete_token(cls):
        """
        Delete token.
        Do not fail if token does not exist.
        """
        ...
    


