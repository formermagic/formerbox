"""
This type stub file was generated by pyright.
"""

from collections import OrderedDict
from .configuration_auto import AlbertConfig, BartConfig, BertConfig, BertGenerationConfig, BlenderbotConfig, CTRLConfig, CamembertConfig, DPRConfig, DebertaConfig, DistilBertConfig, ElectraConfig, FSMTConfig, FlaubertConfig, FunnelConfig, GPT2Config, LayoutLMConfig, LongformerConfig, LxmertConfig, MBartConfig, MarianConfig, MobileBertConfig, OpenAIGPTConfig, PegasusConfig, ProphetNetConfig, RagConfig, ReformerConfig, RetriBertConfig, RobertaConfig, SqueezeBertConfig, T5Config, TransfoXLConfig, XLMConfig, XLMProphetNetConfig, XLMRobertaConfig, XLNetConfig, replace_list_option_in_docstrings
from .file_utils import is_sentencepiece_available, is_tokenizers_available
from .tokenization_bart import BartTokenizer
from .tokenization_bert import BertTokenizer
from .tokenization_bertweet import BertweetTokenizer
from .tokenization_blenderbot import BlenderbotSmallTokenizer
from .tokenization_ctrl import CTRLTokenizer
from .tokenization_deberta import DebertaTokenizer
from .tokenization_distilbert import DistilBertTokenizer
from .tokenization_dpr import DPRQuestionEncoderTokenizer
from .tokenization_electra import ElectraTokenizer
from .tokenization_flaubert import FlaubertTokenizer
from .tokenization_fsmt import FSMTTokenizer
from .tokenization_funnel import FunnelTokenizer
from .tokenization_gpt2 import GPT2Tokenizer
from .tokenization_layoutlm import LayoutLMTokenizer
from .tokenization_longformer import LongformerTokenizer
from .tokenization_lxmert import LxmertTokenizer
from .tokenization_mobilebert import MobileBertTokenizer
from .tokenization_openai import OpenAIGPTTokenizer
from .tokenization_phobert import PhobertTokenizer
from .tokenization_prophetnet import ProphetNetTokenizer
from .tokenization_rag import RagTokenizer
from .tokenization_retribert import RetriBertTokenizer
from .tokenization_roberta import RobertaTokenizer
from .tokenization_squeezebert import SqueezeBertTokenizer
from .tokenization_transfo_xl import TransfoXLTokenizer
from .tokenization_xlm import XLMTokenizer
from .utils import logging
from .tokenization_albert import AlbertTokenizer
from .tokenization_bert_generation import BertGenerationTokenizer
from .tokenization_camembert import CamembertTokenizer
from .tokenization_marian import MarianTokenizer
from .tokenization_mbart import MBartTokenizer
from .tokenization_pegasus import PegasusTokenizer
from .tokenization_reformer import ReformerTokenizer
from .tokenization_t5 import T5Tokenizer
from .tokenization_xlm_prophetnet import XLMProphetNetTokenizer
from .tokenization_xlm_roberta import XLMRobertaTokenizer
from .tokenization_xlnet import XLNetTokenizer
from .tokenization_albert_fast import AlbertTokenizerFast
from .tokenization_bart_fast import BartTokenizerFast
from .tokenization_bert_fast import BertTokenizerFast
from .tokenization_camembert_fast import CamembertTokenizerFast
from .tokenization_distilbert_fast import DistilBertTokenizerFast
from .tokenization_dpr_fast import DPRQuestionEncoderTokenizerFast
from .tokenization_electra_fast import ElectraTokenizerFast
from .tokenization_funnel_fast import FunnelTokenizerFast
from .tokenization_gpt2_fast import GPT2TokenizerFast
from .tokenization_layoutlm_fast import LayoutLMTokenizerFast
from .tokenization_longformer_fast import LongformerTokenizerFast
from .tokenization_lxmert_fast import LxmertTokenizerFast
from .tokenization_mbart_fast import MBartTokenizerFast
from .tokenization_mobilebert_fast import MobileBertTokenizerFast
from .tokenization_openai_fast import OpenAIGPTTokenizerFast
from .tokenization_pegasus_fast import PegasusTokenizerFast
from .tokenization_reformer_fast import ReformerTokenizerFast
from .tokenization_retribert_fast import RetriBertTokenizerFast
from .tokenization_roberta_fast import RobertaTokenizerFast
from .tokenization_squeezebert_fast import SqueezeBertTokenizerFast
from .tokenization_t5_fast import T5TokenizerFast
from .tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast
from .tokenization_xlnet_fast import XLNetTokenizerFast

""" Auto Tokenizer class. """
if is_sentencepiece_available():
    ...
else:
    AlbertTokenizer = None
    BertGenerationTokenizer = None
    CamembertTokenizer = None
    MarianTokenizer = None
    MBartTokenizer = None
    PegasusTokenizer = None
    ReformerTokenizer = None
    T5Tokenizer = None
    XLMRobertaTokenizer = None
    XLNetTokenizer = None
if is_tokenizers_available():
    ...
else:
    AlbertTokenizerFast = None
    BartTokenizerFast = None
    BertTokenizerFast = None
    CamembertTokenizerFast = None
    DistilBertTokenizerFast = None
    DPRQuestionEncoderTokenizerFast = None
    ElectraTokenizerFast = None
    FunnelTokenizerFast = None
    GPT2TokenizerFast = None
    LayoutLMTokenizerFast = None
    LongformerTokenizerFast = None
    LxmertTokenizerFast = None
    MBartTokenizerFast = None
    MobileBertTokenizerFast = None
    OpenAIGPTTokenizerFast = None
    PegasusTokenizerFast = None
    ReformerTokenizerFast = None
    RetriBertTokenizerFast = None
    RobertaTokenizerFast = None
    SqueezeBertTokenizerFast = None
    T5TokenizerFast = None
    XLMRobertaTokenizerFast = None
    XLNetTokenizerFast = None
logger = logging.get_logger(__name__)
TOKENIZER_MAPPING = OrderedDict([(RetriBertConfig, (RetriBertTokenizer, RetriBertTokenizerFast)), (T5Config, (T5Tokenizer, T5TokenizerFast)), (MobileBertConfig, (MobileBertTokenizer, MobileBertTokenizerFast)), (DistilBertConfig, (DistilBertTokenizer, DistilBertTokenizerFast)), (AlbertConfig, (AlbertTokenizer, AlbertTokenizerFast)), (CamembertConfig, (CamembertTokenizer, CamembertTokenizerFast)), (PegasusConfig, (PegasusTokenizer, PegasusTokenizerFast)), (MBartConfig, (MBartTokenizer, MBartTokenizerFast)), (XLMRobertaConfig, (XLMRobertaTokenizer, XLMRobertaTokenizerFast)), (MarianConfig, (MarianTokenizer, None)), (BlenderbotConfig, (BlenderbotSmallTokenizer, None)), (LongformerConfig, (LongformerTokenizer, LongformerTokenizerFast)), (BartConfig, (BartTokenizer, BartTokenizerFast)), (LongformerConfig, (LongformerTokenizer, LongformerTokenizerFast)), (RobertaConfig, (BertweetTokenizer, None)), (RobertaConfig, (PhobertTokenizer, None)), (RobertaConfig, (RobertaTokenizer, RobertaTokenizerFast)), (ReformerConfig, (ReformerTokenizer, ReformerTokenizerFast)), (ElectraConfig, (ElectraTokenizer, ElectraTokenizerFast)), (FunnelConfig, (FunnelTokenizer, FunnelTokenizerFast)), (LxmertConfig, (LxmertTokenizer, LxmertTokenizerFast)), (LayoutLMConfig, (LayoutLMTokenizer, LayoutLMTokenizerFast)), (DPRConfig, (DPRQuestionEncoderTokenizer, DPRQuestionEncoderTokenizerFast)), (SqueezeBertConfig, (SqueezeBertTokenizer, SqueezeBertTokenizerFast)), (BertConfig, (BertTokenizer, BertTokenizerFast)), (OpenAIGPTConfig, (OpenAIGPTTokenizer, OpenAIGPTTokenizerFast)), (GPT2Config, (GPT2Tokenizer, GPT2TokenizerFast)), (TransfoXLConfig, (TransfoXLTokenizer, None)), (XLNetConfig, (XLNetTokenizer, XLNetTokenizerFast)), (FlaubertConfig, (FlaubertTokenizer, None)), (XLMConfig, (XLMTokenizer, None)), (CTRLConfig, (CTRLTokenizer, None)), (FSMTConfig, (FSMTTokenizer, None)), (BertGenerationConfig, (BertGenerationTokenizer, None)), (DebertaConfig, (DebertaTokenizer, None)), (RagConfig, (RagTokenizer, None)), (XLMProphetNetConfig, (XLMProphetNetTokenizer, None)), (ProphetNetConfig, (ProphetNetTokenizer, None))])
SLOW_TOKENIZER_MAPPING = { k: v[0] if v[0] is not None else v[1] for (k, v) in TOKENIZER_MAPPING.items() if v[0] is not None or v[1] is not None }
class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library
    when created with the :meth:`AutoTokenizer.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """
    def __init__(self) -> None:
        ...
    
    @classmethod
    @replace_list_option_in_docstrings(SLOW_TOKENIZER_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r"""
        Instantiate one of the tokenizer classes of the library from a pretrained model vocabulary.

        The tokenizer class to instantiate is selected based on the :obj:`model_type` property of the config object
        (either passed as an argument or loaded from :obj:`pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on :obj:`pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (:obj:`str`):
                Can be either:

                    - A string with the `shortcut name` of a predefined tokenizer to load from cache or download, e.g.,
                      ``bert-base-uncased``.
                    - A string with the `identifier name` of a predefined tokenizer that was user-uploaded to our S3,
                      e.g., ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing vocabulary files required by the tokenizer, for instance saved
                      using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.,
                      ``./my_model_directory/``.
                    - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: ``./my_model_directory/vocab.txt``.
                      (Not applicable to all derived classes)
            inputs (additional positional arguments, `optional`):
                Will be passed along to the Tokenizer ``__init__()`` method.
            config (:class:`~transformers.PreTrainedConfig`, `optional`)
                The configuration object used to dertermine the tokenizer class to instantiate.
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g.,
                :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each
                request.
            use_fast (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to try to load the fast version of the tokenizer.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the Tokenizer ``__init__()`` method. Can be used to set special tokens like
                ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``,
                ``mask_token``, ``additional_special_tokens``. See parameters in the ``__init__()`` for more details.

        Examples::

            >>> from transformers import AutoTokenizer

            >>> # Download vocabulary from S3 and cache.
            >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            >>> # Download vocabulary from S3 (user-uploaded) and cache.
            >>> tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')

            >>> # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
            >>> tokenizer = AutoTokenizer.from_pretrained('./test/bert_saved_model/')

        """
        ...
    


