try:
    from importlib.metadata import PackageNotFoundError, version  # type: ignore
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

from formerbox.cli import ConvertDataset, Preprocess, Subcommand, Train, TrainTokenizer
from formerbox.common import (
    DataclassArgumentParser,
    DataclassBase,
    FromPartial,
    HasParams,
    HasParsableParams,
    Registrable,
)
from formerbox.data import (
    BartTokenizer,
    BartTokenizerTrainer,
    Binarizer,
    DatasetConverter,
    DatasetIterator,
    DefaultBinarizer,
    DefaultDatasetConverter,
    GPT2Tokenizer,
    GPT2TokenizerTrainer,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedDatasetBuilder,
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
    RobertaTokenizer,
    RobertaTokenizerTrainer,
    TokenizerBase,
    TokenizerTrainerBase,
    TranslationBinarizer,
    TranslationDataset,
)
from formerbox.modules import (
    DenoisingDataModule,
    DenoisingModule,
    MaskedLMDataModule,
    MaskedLMModule,
    TokenizerTrainer,
    TransformerDataModule,
    TransformerModule,
    TranslationDataModule,
    TranslationModule,
    WordLMDataModule,
    WordLMModule,
)
from formerbox.optim import AdamW, get_polynomial_decay_with_warmup, weight_decay_params
from formerbox.tasks import (
    CodeDatasetConverter,
    CodeRobertaTokenizer,
    CodeRobertaTokenizerTrainer,
    DenoisingTask,
    MaskedLMTask,
    TaskModule,
    TranslationTask,
    WordLMTask,
)
from formerbox.training import TransformerTrainer

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
