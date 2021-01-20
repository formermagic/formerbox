"""
This type stub file was generated by pyright.
"""

from collections import OrderedDict
from ..albert.configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from ..bart.configuration_bart import BART_PRETRAINED_CONFIG_ARCHIVE_MAP, BartConfig
from ..bert.configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from ..bert_generation.configuration_bert_generation import BertGenerationConfig
from ..blenderbot.configuration_blenderbot import BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP, BlenderbotConfig
from ..blenderbot_small.configuration_blenderbot_small import BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP, BlenderbotSmallConfig
from ..camembert.configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from ..ctrl.configuration_ctrl import CTRLConfig, CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP
from ..deberta.configuration_deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaConfig
from ..distilbert.configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
from ..dpr.configuration_dpr import DPRConfig, DPR_PRETRAINED_CONFIG_ARCHIVE_MAP
from ..electra.configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
from ..encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from ..flaubert.configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
from ..fsmt.configuration_fsmt import FSMTConfig, FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP
from ..funnel.configuration_funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig
from ..gpt2.configuration_gpt2 import GPT2Config, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP
from ..layoutlm.configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig
from ..led.configuration_led import LEDConfig, LED_PRETRAINED_CONFIG_ARCHIVE_MAP
from ..longformer.configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
from ..lxmert.configuration_lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig
from ..marian.configuration_marian import MarianConfig
from ..mbart.configuration_mbart import MBART_PRETRAINED_CONFIG_ARCHIVE_MAP, MBartConfig
from ..mobilebert.configuration_mobilebert import MobileBertConfig
from ..mpnet.configuration_mpnet import MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP, MPNetConfig
from ..mt5.configuration_mt5 import MT5Config
from ..openai.configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from ..pegasus.configuration_pegasus import PegasusConfig
from ..prophetnet.configuration_prophetnet import PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ProphetNetConfig
from ..rag.configuration_rag import RagConfig
from ..reformer.configuration_reformer import ReformerConfig
from ..retribert.configuration_retribert import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RetriBertConfig
from ..roberta.configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
from ..squeezebert.configuration_squeezebert import SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, SqueezeBertConfig
from ..t5.configuration_t5 import T5Config, T5_PRETRAINED_CONFIG_ARCHIVE_MAP
from ..tapas.configuration_tapas import TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP, TapasConfig
from ..transfo_xl.configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from ..xlm.configuration_xlm import XLMConfig, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP
from ..xlm_prophetnet.configuration_xlm_prophetnet import XLMProphetNetConfig, XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP
from ..xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig, XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
from ..xlnet.configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig

""" Auto Config class. """
ALL_PRETRAINED_CONFIG_ARCHIVE_MAP = dict((key, value) for pretrained_map in [LED_PRETRAINED_CONFIG_ARCHIVE_MAP, BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BART_PRETRAINED_CONFIG_ARCHIVE_MAP, BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP, MBART_PRETRAINED_CONFIG_ARCHIVE_MAP, OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, T5_PRETRAINED_CONFIG_ARCHIVE_MAP, XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, DPR_PRETRAINED_CONFIG_ARCHIVE_MAP, DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP, TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP] for (key, value) in pretrained_map.items())
CONFIG_MAPPING = OrderedDict([("led", LEDConfig), ("blenderbot-small", BlenderbotSmallConfig), ("retribert", RetriBertConfig), ("mt5", MT5Config), ("t5", T5Config), ("mobilebert", MobileBertConfig), ("distilbert", DistilBertConfig), ("albert", AlbertConfig), ("bert-generation", BertGenerationConfig), ("camembert", CamembertConfig), ("xlm-roberta", XLMRobertaConfig), ("pegasus", PegasusConfig), ("marian", MarianConfig), ("mbart", MBartConfig), ("mpnet", MPNetConfig), ("bart", BartConfig), ("blenderbot", BlenderbotConfig), ("reformer", ReformerConfig), ("longformer", LongformerConfig), ("roberta", RobertaConfig), ("deberta", DebertaConfig), ("flaubert", FlaubertConfig), ("fsmt", FSMTConfig), ("squeezebert", SqueezeBertConfig), ("bert", BertConfig), ("openai-gpt", OpenAIGPTConfig), ("gpt2", GPT2Config), ("transfo-xl", TransfoXLConfig), ("xlnet", XLNetConfig), ("xlm-prophetnet", XLMProphetNetConfig), ("prophetnet", ProphetNetConfig), ("xlm", XLMConfig), ("ctrl", CTRLConfig), ("electra", ElectraConfig), ("encoder-decoder", EncoderDecoderConfig), ("funnel", FunnelConfig), ("lxmert", LxmertConfig), ("dpr", DPRConfig), ("layoutlm", LayoutLMConfig), ("rag", RagConfig), ("tapas", TapasConfig)])
MODEL_NAMES_MAPPING = OrderedDict([("led", "LED"), ("blenderbot-small", "BlenderbotSmall"), ("retribert", "RetriBERT"), ("t5", "T5"), ("mobilebert", "MobileBERT"), ("distilbert", "DistilBERT"), ("albert", "ALBERT"), ("bert-generation", "Bert Generation"), ("camembert", "CamemBERT"), ("xlm-roberta", "XLM-RoBERTa"), ("pegasus", "Pegasus"), ("blenderbot", "Blenderbot"), ("marian", "Marian"), ("mbart", "mBART"), ("bart", "BART"), ("reformer", "Reformer"), ("longformer", "Longformer"), ("roberta", "RoBERTa"), ("flaubert", "FlauBERT"), ("fsmt", "FairSeq Machine-Translation"), ("squeezebert", "SqueezeBERT"), ("bert", "BERT"), ("openai-gpt", "OpenAI GPT"), ("gpt2", "OpenAI GPT-2"), ("transfo-xl", "Transformer-XL"), ("xlnet", "XLNet"), ("xlm", "XLM"), ("ctrl", "CTRL"), ("electra", "ELECTRA"), ("encoder-decoder", "Encoder decoder"), ("funnel", "Funnel Transformer"), ("lxmert", "LXMERT"), ("deberta", "DeBERTa"), ("layoutlm", "LayoutLM"), ("dpr", "DPR"), ("rag", "RAG"), ("xlm-prophetnet", "XLMProphetNet"), ("prophetnet", "ProphetNet"), ("mt5", "mT5"), ("mpnet", "MPNet"), ("tapas", "TAPAS")])
def replace_list_option_in_docstrings(config_to_class=..., use_model_types=...):
    ...

class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the :meth:`~transformers.AutoConfig.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """
    def __init__(self) -> None:
        ...
    
    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        ...
    
    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the :obj:`model_type` property of the config object
        that is loaded, or when it's missing, by falling back to using pattern matching on
        :obj:`pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a pretrained model configuration hosted inside a model repo on
                      huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                      namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing a configuration file saved using the
                      :meth:`~transformers.PretrainedConfig.save_pretrained` method, or the
                      :meth:`~transformers.PreTrainedModel.save_pretrained` method, e.g., ``./my_model_directory/``.
                    - A path or url to a saved configuration JSON `file`, e.g.,
                      ``./my_model_directory/configuration.json``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final configuration object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e.,
                the part of ``kwargs`` which has not been used to update ``config`` and is otherwise ignored.
            kwargs(additional keyword arguments, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the ``return_unused_kwargs`` keyword parameter.

        Examples::

            >>> from transformers import AutoConfig

            >>> # Download configuration from huggingface.co and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')

            >>> # Download configuration from huggingface.co (user-uploaded) and cache.
            >>> config = AutoConfig.from_pretrained('dbmdz/bert-base-german-cased')

            >>> # If configuration file is in a directory (e.g., was saved using `save_pretrained('./test/saved_model/')`).
            >>> config = AutoConfig.from_pretrained('./test/bert_saved_model/')

            >>> # Load a specific configuration file.
            >>> config = AutoConfig.from_pretrained('./test/bert_saved_model/my_configuration.json')

            >>> # Change some config attributes when loading a pretrained config.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False)
            >>> config.output_attentions
            True
            >>> config, unused_kwargs = AutoConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False, return_unused_kwargs=True)
            >>> config.output_attentions
            True
            >>> config.unused_kwargs
            {'foo': False}
        """
        ...
    


