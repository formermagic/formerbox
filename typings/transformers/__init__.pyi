"""
This type stub file was generated by pyright.
"""

from .integrations import is_comet_available, is_optuna_available, is_ray_available, is_tensorboard_available, is_wandb_available
from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, AutoConfig, CONFIG_MAPPING
from .configuration_bart import BartConfig
from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from .configuration_bert_generation import BertGenerationConfig
from .configuration_blenderbot import BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP, BlenderbotConfig
from .configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from .configuration_ctrl import CTRLConfig, CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaConfig
from .configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
from .configuration_dpr import DPRConfig, DPR_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
from .configuration_encoder_decoder import EncoderDecoderConfig
from .configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
from .configuration_fsmt import FSMTConfig, FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig
from .configuration_gpt2 import GPT2Config, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig
from .configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
from .configuration_lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig
from .configuration_marian import MarianConfig
from .configuration_mbart import MBartConfig
from .configuration_mmbt import MMBTConfig
from .configuration_mobilebert import MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileBertConfig
from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from .configuration_pegasus import PegasusConfig
from .configuration_prophetnet import PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ProphetNetConfig
from .configuration_rag import RagConfig
from .configuration_reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig
from .configuration_retribert import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RetriBertConfig
from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
from .configuration_squeezebert import SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, SqueezeBertConfig
from .configuration_t5 import T5Config, T5_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from .configuration_utils import PretrainedConfig
from .configuration_xlm import XLMConfig, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_xlm_prophetnet import XLMProphetNetConfig, XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_xlm_roberta import XLMRobertaConfig, XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig
from .data import DataProcessor, InputExample, InputFeatures, SingleSentenceClassificationProcessor, SquadExample, SquadFeatures, SquadV1Processor, SquadV2Processor, glue_compute_metrics, glue_convert_examples_to_features, glue_output_modes, glue_processors, glue_tasks_num_labels, squad_convert_examples_to_features, xnli_compute_metrics, xnli_output_modes, xnli_processors, xnli_tasks_num_labels
from .file_utils import CONFIG_NAME, MODEL_CARD_NAME, PYTORCH_PRETRAINED_BERT_CACHE, PYTORCH_TRANSFORMERS_CACHE, SPIECE_UNDERLINE, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, TRANSFORMERS_CACHE, WEIGHTS_NAME, add_end_docstrings, add_start_docstrings, cached_path, is_apex_available, is_datasets_available, is_faiss_available, is_flax_available, is_psutil_available, is_py3nvml_available, is_sentencepiece_available, is_sklearn_available, is_tf_available, is_tokenizers_available, is_torch_available, is_torch_tpu_available
from .hf_argparser import HfArgumentParser
from .modelcard import ModelCard
from .modeling_tf_pytorch_utils import convert_tf_weight_name_to_pt_weight_name, load_pytorch_checkpoint_in_tf2_model, load_pytorch_model_in_tf2_model, load_pytorch_weights_in_tf2_model, load_tf2_checkpoint_in_pytorch_model, load_tf2_model_in_pytorch_model, load_tf2_weights_in_pytorch_model
from .pipelines import Conversation, ConversationalPipeline, CsvPipelineDataFormat, FeatureExtractionPipeline, FillMaskPipeline, JsonPipelineDataFormat, NerPipeline, PipedPipelineDataFormat, Pipeline, PipelineDataFormat, QuestionAnsweringPipeline, SummarizationPipeline, Text2TextGenerationPipeline, TextClassificationPipeline, TextGenerationPipeline, TokenClassificationPipeline, TranslationPipeline, ZeroShotClassificationPipeline, pipeline
from .retrieval_rag import RagRetriever
from .tokenization_auto import AutoTokenizer, TOKENIZER_MAPPING
from .tokenization_bart import BartTokenizer
from .tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer
from .tokenization_bert_japanese import BertJapaneseTokenizer, CharacterTokenizer, MecabTokenizer
from .tokenization_bertweet import BertweetTokenizer
from .tokenization_blenderbot import BlenderbotSmallTokenizer, BlenderbotTokenizer
from .tokenization_ctrl import CTRLTokenizer
from .tokenization_deberta import DebertaTokenizer
from .tokenization_distilbert import DistilBertTokenizer
from .tokenization_dpr import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRReaderOutput, DPRReaderTokenizer
from .tokenization_electra import ElectraTokenizer
from .tokenization_flaubert import FlaubertTokenizer
from .tokenization_fsmt import FSMTTokenizer
from .tokenization_funnel import FunnelTokenizer
from .tokenization_gpt2 import GPT2Tokenizer
from .tokenization_herbert import HerbertTokenizer
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
from .tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import AddedToken, BatchEncoding, CharSpan, PreTrainedTokenizerBase, SpecialTokensMixin, TensorType, TokenSpan
from .tokenization_xlm import XLMTokenizer
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
from .utils.dummy_sentencepiece_objects import *
from .tokenization_albert_fast import AlbertTokenizerFast
from .tokenization_bart_fast import BartTokenizerFast
from .tokenization_bert_fast import BertTokenizerFast
from .tokenization_camembert_fast import CamembertTokenizerFast
from .tokenization_distilbert_fast import DistilBertTokenizerFast
from .tokenization_dpr_fast import DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast, DPRReaderTokenizerFast
from .tokenization_electra_fast import ElectraTokenizerFast
from .tokenization_funnel_fast import FunnelTokenizerFast
from .tokenization_gpt2_fast import GPT2TokenizerFast
from .tokenization_herbert_fast import HerbertTokenizerFast
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
from .tokenization_utils_fast import PreTrainedTokenizerFast
from .tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast
from .tokenization_xlnet_fast import XLNetTokenizerFast
from .utils.dummy_tokenizers_objects import *
from .trainer_callback import DefaultFlowCallback, PrinterCallback, ProgressCallback, TrainerCallback, TrainerControl, TrainerState
from .trainer_utils import EvalPrediction, EvaluationStrategy, set_seed
from .training_args import TrainingArguments
from .training_args_tf import TFTrainingArguments
from .utils import logging
from .benchmark.benchmark import PyTorchBenchmark
from .benchmark.benchmark_args import PyTorchBenchmarkArguments
from .data.data_collator import DataCollator, DataCollatorForLanguageModeling, DataCollatorForNextSentencePrediction, DataCollatorForPermutationLanguageModeling, DataCollatorForSOP, DataCollatorWithPadding, default_data_collator
from .data.datasets import GlueDataTrainingArguments, GlueDataset, LineByLineTextDataset, LineByLineWithSOPTextDataset, SquadDataTrainingArguments, SquadDataset, TextDataset, TextDatasetForNextSentencePrediction
from .generation_utils import top_k_top_p_filtering
from .modeling_albert import ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST, AlbertForMaskedLM, AlbertForMultipleChoice, AlbertForPreTraining, AlbertForQuestionAnswering, AlbertForSequenceClassification, AlbertForTokenClassification, AlbertModel, AlbertPreTrainedModel, load_tf_weights_in_albert
from .modeling_auto import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForMultipleChoice, AutoModelForPreTraining, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelWithLMHead, MODEL_FOR_CAUSAL_LM_MAPPING, MODEL_FOR_MASKED_LM_MAPPING, MODEL_FOR_MULTIPLE_CHOICE_MAPPING, MODEL_FOR_PRETRAINING_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_MAPPING, MODEL_WITH_LM_HEAD_MAPPING
from .modeling_bart import BART_PRETRAINED_MODEL_ARCHIVE_LIST, BartForConditionalGeneration, BartForQuestionAnswering, BartForSequenceClassification, BartModel, PretrainedBartModel
from .modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_LIST, BertForMaskedLM, BertForMultipleChoice, BertForNextSentencePrediction, BertForPreTraining, BertForQuestionAnswering, BertForSequenceClassification, BertForTokenClassification, BertLMHeadModel, BertLayer, BertModel, BertPreTrainedModel, load_tf_weights_in_bert
from .modeling_bert_generation import BertGenerationDecoder, BertGenerationEncoder, load_tf_weights_in_bert_generation
from .modeling_blenderbot import BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST, BlenderbotForConditionalGeneration
from .modeling_camembert import CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST, CamembertForCausalLM, CamembertForMaskedLM, CamembertForMultipleChoice, CamembertForQuestionAnswering, CamembertForSequenceClassification, CamembertForTokenClassification, CamembertModel
from .modeling_ctrl import CTRLLMHeadModel, CTRLModel, CTRLPreTrainedModel, CTRL_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_deberta import DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, DebertaForSequenceClassification, DebertaModel, DebertaPreTrainedModel
from .modeling_distilbert import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST, DistilBertForMaskedLM, DistilBertForMultipleChoice, DistilBertForQuestionAnswering, DistilBertForSequenceClassification, DistilBertForTokenClassification, DistilBertModel, DistilBertPreTrainedModel
from .modeling_dpr import DPRContextEncoder, DPRPretrainedContextEncoder, DPRPretrainedQuestionEncoder, DPRPretrainedReader, DPRQuestionEncoder, DPRReader
from .modeling_electra import ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST, ElectraForMaskedLM, ElectraForMultipleChoice, ElectraForPreTraining, ElectraForQuestionAnswering, ElectraForSequenceClassification, ElectraForTokenClassification, ElectraModel, ElectraPreTrainedModel, load_tf_weights_in_electra
from .modeling_encoder_decoder import EncoderDecoderModel
from .modeling_flaubert import FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST, FlaubertForMultipleChoice, FlaubertForQuestionAnswering, FlaubertForQuestionAnsweringSimple, FlaubertForSequenceClassification, FlaubertForTokenClassification, FlaubertModel, FlaubertWithLMHeadModel
from .modeling_fsmt import FSMTForConditionalGeneration, FSMTModel, PretrainedFSMTModel
from .modeling_funnel import FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST, FunnelBaseModel, FunnelForMaskedLM, FunnelForMultipleChoice, FunnelForPreTraining, FunnelForQuestionAnswering, FunnelForSequenceClassification, FunnelForTokenClassification, FunnelModel, load_tf_weights_in_funnel
from .modeling_gpt2 import GPT2DoubleHeadsModel, GPT2ForSequenceClassification, GPT2LMHeadModel, GPT2Model, GPT2PreTrainedModel, GPT2_PRETRAINED_MODEL_ARCHIVE_LIST, load_tf_weights_in_gpt2
from .modeling_layoutlm import LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST, LayoutLMForMaskedLM, LayoutLMForTokenClassification, LayoutLMModel
from .modeling_longformer import LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST, LongformerForMaskedLM, LongformerForMultipleChoice, LongformerForQuestionAnswering, LongformerForSequenceClassification, LongformerForTokenClassification, LongformerModel, LongformerSelfAttention
from .modeling_lxmert import LxmertEncoder, LxmertForPreTraining, LxmertForQuestionAnswering, LxmertModel, LxmertPreTrainedModel, LxmertVisualFeatureEncoder, LxmertXLayer
from .modeling_marian import MarianMTModel
from .modeling_mbart import MBartForConditionalGeneration
from .modeling_mmbt import MMBTForClassification, MMBTModel, ModalEmbeddings
from .modeling_mobilebert import MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST, MobileBertForMaskedLM, MobileBertForMultipleChoice, MobileBertForNextSentencePrediction, MobileBertForPreTraining, MobileBertForQuestionAnswering, MobileBertForSequenceClassification, MobileBertForTokenClassification, MobileBertLayer, MobileBertModel, MobileBertPreTrainedModel, load_tf_weights_in_mobilebert
from .modeling_openai import OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST, OpenAIGPTDoubleHeadsModel, OpenAIGPTForSequenceClassification, OpenAIGPTLMHeadModel, OpenAIGPTModel, OpenAIGPTPreTrainedModel, load_tf_weights_in_openai_gpt
from .modeling_pegasus import PegasusForConditionalGeneration
from .modeling_prophetnet import PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST, ProphetNetDecoder, ProphetNetEncoder, ProphetNetForCausalLM, ProphetNetForConditionalGeneration, ProphetNetModel, ProphetNetPreTrainedModel
from .modeling_rag import RagModel, RagSequenceForGeneration, RagTokenForGeneration
from .modeling_reformer import REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST, ReformerAttention, ReformerForMaskedLM, ReformerForQuestionAnswering, ReformerForSequenceClassification, ReformerLayer, ReformerModel, ReformerModelWithLMHead
from .modeling_retribert import RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST, RetriBertModel, RetriBertPreTrainedModel
from .modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, RobertaForCausalLM, RobertaForMaskedLM, RobertaForMultipleChoice, RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaForTokenClassification, RobertaModel
from .modeling_squeezebert import SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST, SqueezeBertForMaskedLM, SqueezeBertForMultipleChoice, SqueezeBertForQuestionAnswering, SqueezeBertForSequenceClassification, SqueezeBertForTokenClassification, SqueezeBertModel, SqueezeBertModule, SqueezeBertPreTrainedModel
from .modeling_t5 import T5ForConditionalGeneration, T5Model, T5PreTrainedModel, T5_PRETRAINED_MODEL_ARCHIVE_LIST, load_tf_weights_in_t5
from .modeling_transfo_xl import AdaptiveEmbedding, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST, TransfoXLLMHeadModel, TransfoXLModel, TransfoXLPreTrainedModel, load_tf_weights_in_transfo_xl
from .modeling_utils import Conv1D, PreTrainedModel, apply_chunking_to_forward, prune_layer
from .modeling_xlm import XLMForMultipleChoice, XLMForQuestionAnswering, XLMForQuestionAnsweringSimple, XLMForSequenceClassification, XLMForTokenClassification, XLMModel, XLMPreTrainedModel, XLMWithLMHeadModel, XLM_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_xlm_prophetnet import XLMProphetNetDecoder, XLMProphetNetEncoder, XLMProphetNetForCausalLM, XLMProphetNetForConditionalGeneration, XLMProphetNetModel, XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_xlm_roberta import XLMRobertaForCausalLM, XLMRobertaForMaskedLM, XLMRobertaForMultipleChoice, XLMRobertaForQuestionAnswering, XLMRobertaForSequenceClassification, XLMRobertaForTokenClassification, XLMRobertaModel, XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_xlnet import XLNET_PRETRAINED_MODEL_ARCHIVE_LIST, XLNetForMultipleChoice, XLNetForQuestionAnswering, XLNetForQuestionAnsweringSimple, XLNetForSequenceClassification, XLNetForTokenClassification, XLNetLMHeadModel, XLNetModel, XLNetPreTrainedModel, load_tf_weights_in_xlnet
from .optimization import Adafactor, AdamW, get_constant_schedule, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from .trainer import Trainer
from .trainer_pt_utils import torch_distributed_zero_first
from .utils.dummy_pt_objects import *
from .benchmark.benchmark_args_tf import TensorFlowBenchmarkArguments
from .benchmark.benchmark_tf import TensorFlowBenchmark
from .generation_tf_utils import tf_top_k_top_p_filtering
from .modeling_tf_albert import TFAlbertForMaskedLM, TFAlbertForMultipleChoice, TFAlbertForPreTraining, TFAlbertForQuestionAnswering, TFAlbertForSequenceClassification, TFAlbertForTokenClassification, TFAlbertMainLayer, TFAlbertModel, TFAlbertPreTrainedModel, TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_auto import TFAutoModel, TFAutoModelForCausalLM, TFAutoModelForMaskedLM, TFAutoModelForMultipleChoice, TFAutoModelForPreTraining, TFAutoModelForQuestionAnswering, TFAutoModelForSeq2SeqLM, TFAutoModelForSequenceClassification, TFAutoModelForTokenClassification, TFAutoModelWithLMHead, TF_MODEL_FOR_CAUSAL_LM_MAPPING, TF_MODEL_FOR_MASKED_LM_MAPPING, TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING, TF_MODEL_FOR_PRETRAINING_MAPPING, TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING, TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, TF_MODEL_MAPPING, TF_MODEL_WITH_LM_HEAD_MAPPING
from .modeling_tf_bert import TFBertEmbeddings, TFBertForMaskedLM, TFBertForMultipleChoice, TFBertForNextSentencePrediction, TFBertForPreTraining, TFBertForQuestionAnswering, TFBertForSequenceClassification, TFBertForTokenClassification, TFBertLMHeadModel, TFBertMainLayer, TFBertModel, TFBertPreTrainedModel, TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_camembert import TFCamembertForMaskedLM, TFCamembertForMultipleChoice, TFCamembertForQuestionAnswering, TFCamembertForSequenceClassification, TFCamembertForTokenClassification, TFCamembertModel, TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_ctrl import TFCTRLLMHeadModel, TFCTRLModel, TFCTRLPreTrainedModel, TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_distilbert import TFDistilBertForMaskedLM, TFDistilBertForMultipleChoice, TFDistilBertForQuestionAnswering, TFDistilBertForSequenceClassification, TFDistilBertForTokenClassification, TFDistilBertMainLayer, TFDistilBertModel, TFDistilBertPreTrainedModel, TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_electra import TFElectraForMaskedLM, TFElectraForMultipleChoice, TFElectraForPreTraining, TFElectraForQuestionAnswering, TFElectraForSequenceClassification, TFElectraForTokenClassification, TFElectraModel, TFElectraPreTrainedModel, TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_flaubert import TFFlaubertForMultipleChoice, TFFlaubertForQuestionAnsweringSimple, TFFlaubertForSequenceClassification, TFFlaubertForTokenClassification, TFFlaubertModel, TFFlaubertWithLMHeadModel, TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_funnel import TFFunnelBaseModel, TFFunnelForMaskedLM, TFFunnelForMultipleChoice, TFFunnelForPreTraining, TFFunnelForQuestionAnswering, TFFunnelForSequenceClassification, TFFunnelForTokenClassification, TFFunnelModel, TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_gpt2 import TFGPT2DoubleHeadsModel, TFGPT2LMHeadModel, TFGPT2MainLayer, TFGPT2Model, TFGPT2PreTrainedModel, TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_longformer import TFLongformerForMaskedLM, TFLongformerForQuestionAnswering, TFLongformerModel, TFLongformerSelfAttention, TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_lxmert import TFLxmertForPreTraining, TFLxmertMainLayer, TFLxmertModel, TFLxmertPreTrainedModel, TFLxmertVisualFeatureEncoder, TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_mobilebert import TFMobileBertForMaskedLM, TFMobileBertForMultipleChoice, TFMobileBertForNextSentencePrediction, TFMobileBertForPreTraining, TFMobileBertForQuestionAnswering, TFMobileBertForSequenceClassification, TFMobileBertForTokenClassification, TFMobileBertMainLayer, TFMobileBertModel, TFMobileBertPreTrainedModel, TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_openai import TFOpenAIGPTDoubleHeadsModel, TFOpenAIGPTLMHeadModel, TFOpenAIGPTMainLayer, TFOpenAIGPTModel, TFOpenAIGPTPreTrainedModel, TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_roberta import TFRobertaForMaskedLM, TFRobertaForMultipleChoice, TFRobertaForQuestionAnswering, TFRobertaForSequenceClassification, TFRobertaForTokenClassification, TFRobertaMainLayer, TFRobertaModel, TFRobertaPreTrainedModel, TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_t5 import TFT5ForConditionalGeneration, TFT5Model, TFT5PreTrainedModel, TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_transfo_xl import TFAdaptiveEmbedding, TFTransfoXLLMHeadModel, TFTransfoXLMainLayer, TFTransfoXLModel, TFTransfoXLPreTrainedModel, TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_utils import TFPreTrainedModel, TFSequenceSummary, TFSharedEmbeddings, shape_list
from .modeling_tf_xlm import TFXLMForMultipleChoice, TFXLMForQuestionAnsweringSimple, TFXLMForSequenceClassification, TFXLMForTokenClassification, TFXLMMainLayer, TFXLMModel, TFXLMPreTrainedModel, TFXLMWithLMHeadModel, TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_xlm_roberta import TFXLMRobertaForMaskedLM, TFXLMRobertaForMultipleChoice, TFXLMRobertaForQuestionAnswering, TFXLMRobertaForSequenceClassification, TFXLMRobertaForTokenClassification, TFXLMRobertaModel, TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_tf_xlnet import TFXLNetForMultipleChoice, TFXLNetForQuestionAnsweringSimple, TFXLNetForSequenceClassification, TFXLNetForTokenClassification, TFXLNetLMHeadModel, TFXLNetMainLayer, TFXLNetModel, TFXLNetPreTrainedModel, TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST
from .optimization_tf import AdamWeightDecay, GradientAccumulator, WarmUp, create_optimizer
from .trainer_tf import TFTrainer
from .utils.dummy_tf_objects import *
from .modeling_flax_bert import FlaxBertModel
from .modeling_flax_roberta import FlaxRobertaModel
from .utils.dummy_flax_objects import *

__version__ = "3.4.0"
if is_sentencepiece_available():
    ...
else:
    ...
if is_tokenizers_available():
    ...
else:
    ...
logger = logging.get_logger(__name__)
if is_torch_available():
    ...
else:
    ...
if is_tf_available():
    ...
else:
    ...
if is_flax_available():
    ...
else:
    ...
if not is_tf_available() and not is_torch_available():
    ...
