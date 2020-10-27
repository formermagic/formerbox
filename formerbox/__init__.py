try:
    from importlib.metadata import PackageNotFoundError, version  # type: ignore
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

from formerbox.cli import ConvertDataset, Preprocess, Subcommand, Train, TrainTokenizer
from formerbox.common import (
    DataclassArgumentParser,
    DataclassBase,
    HasParams,
    HasParsableParams,
    PartialInitable,
    Registrable,
)
from formerbox.data import (
    Binarizer,
    DatasetConverter,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedDatasetBuilder,
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
    TransformerBinarizer,
    TransformerDatasetConverter,
)
from formerbox.modules import TokenizerModule, TransformerDataModule, TransformerModule
from formerbox.optim import AdamW, get_polynomial_decay_with_warmup, weight_decay_params
from formerbox.tasks import (
    ByteLevelBPETokenizerFast,
    ByteLevelBPETokenizerModule,
    CodeBBPETokenizerFast,
    CodeBBPETokenizerModule,
    CodeDatasetConverter,
    TaskModule,
    TransformerTask,
    TransformerTokenizerModule,
)
from formerbox.training import TransformerTrainer

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
