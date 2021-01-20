"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from . import dependency_versions_check
from .file_utils import CONFIG_NAME, MODEL_CARD_NAME, PYTORCH_PRETRAINED_BERT_CACHE, PYTORCH_TRANSFORMERS_CACHE, SPIECE_UNDERLINE, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, TRANSFORMERS_CACHE, WEIGHTS_NAME, _BaseLazyModule, add_end_docstrings, add_start_docstrings, cached_path, is_apex_available, is_datasets_available, is_faiss_available, is_flax_available, is_psutil_available, is_py3nvml_available, is_sentencepiece_available, is_sklearn_available, is_tf_available, is_tokenizers_available, is_torch_available, is_torch_tpu_available
from .utils import dummy_flax_objects, dummy_pt_objects, dummy_sentencepiece_objects, dummy_tf_objects, dummy_tokenizers_objects, logging
from .configuration_utils import PretrainedConfig
from .data import DataProcessor, InputExample, InputFeatures, SingleSentenceClassificationProcessor, SquadExample, SquadFeatures, SquadV1Processor, SquadV2Processor, glue_compute_metrics, glue_convert_examples_to_features, glue_output_modes, glue_processors, glue_tasks_num_labels, squad_convert_examples_to_features, xnli_compute_metrics, xnli_output_modes, xnli_processors, xnli_tasks_num_labels
from .hf_argparser import HfArgumentParser
from .integrations import is_comet_available, is_optuna_available, is_ray_available, is_ray_tune_available, is_tensorboard_available, is_wandb_available
from .modelcard import ModelCard
from .modeling_tf_pytorch_utils import convert_tf_weight_name_to_pt_weight_name, load_pytorch_checkpoint_in_tf2_model, load_pytorch_model_in_tf2_model, load_pytorch_weights_in_tf2_model, load_tf2_checkpoint_in_pytorch_model, load_tf2_model_in_pytorch_model, load_tf2_weights_in_pytorch_model
from .models.albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST, AlbertConfig, AlbertForMaskedLM, AlbertForMultipleChoice, AlbertForPreTraining, AlbertForQuestionAnswering, AlbertForSequenceClassification, AlbertForTokenClassification, AlbertModel, AlbertPreTrainedModel, AlbertTokenizer, AlbertTokenizerFast, TFAlbertForMaskedLM, TFAlbertForMultipleChoice, TFAlbertForPreTraining, TFAlbertForQuestionAnswering, TFAlbertForSequenceClassification, TFAlbertForTokenClassification, TFAlbertMainLayer, TFAlbertModel, TFAlbertPreTrainedModel, TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST, load_tf_weights_in_albert
from .models.auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForMultipleChoice, AutoModelForNextSentencePrediction, AutoModelForPreTraining, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForTableQuestionAnswering, AutoModelForTokenClassification, AutoModelWithLMHead, AutoTokenizer, CONFIG_MAPPING, FLAX_MODEL_MAPPING, FlaxAutoModel, MODEL_FOR_CAUSAL_LM_MAPPING, MODEL_FOR_MASKED_LM_MAPPING, MODEL_FOR_MULTIPLE_CHOICE_MAPPING, MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING, MODEL_FOR_PRETRAINING_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_MAPPING, MODEL_NAMES_MAPPING, MODEL_WITH_LM_HEAD_MAPPING, TFAutoModel, TFAutoModelForCausalLM, TFAutoModelForMaskedLM, TFAutoModelForMultipleChoice, TFAutoModelForPreTraining, TFAutoModelForQuestionAnswering, TFAutoModelForSeq2SeqLM, TFAutoModelForSequenceClassification, TFAutoModelForTokenClassification, TFAutoModelWithLMHead, TF_MODEL_FOR_CAUSAL_LM_MAPPING, TF_MODEL_FOR_MASKED_LM_MAPPING, TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING, TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING, TF_MODEL_FOR_PRETRAINING_MAPPING, TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING, TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, TF_MODEL_MAPPING, TF_MODEL_WITH_LM_HEAD_MAPPING, TOKENIZER_MAPPING
from .models.bart import BART_PRETRAINED_MODEL_ARCHIVE_LIST, BartConfig, BartForConditionalGeneration, BartForQuestionAnswering, BartForSequenceClassification, BartModel, BartPretrainedModel, BartTokenizer, BartTokenizerFast, PretrainedBartModel, TFBartForConditionalGeneration, TFBartModel, TFBartPretrainedModel
from .models.bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_MODEL_ARCHIVE_LIST, BasicTokenizer, BertConfig, BertForMaskedLM, BertForMultipleChoice, BertForNextSentencePrediction, BertForPreTraining, BertForQuestionAnswering, BertForSequenceClassification, BertForTokenClassification, BertLMHeadModel, BertLayer, BertModel, BertPreTrainedModel, BertTokenizer, BertTokenizerFast, FlaxBertForMaskedLM, FlaxBertModel, TFBertEmbeddings, TFBertForMaskedLM, TFBertForMultipleChoice, TFBertForNextSentencePrediction, TFBertForPreTraining, TFBertForQuestionAnswering, TFBertForSequenceClassification, TFBertForTokenClassification, TFBertLMHeadModel, TFBertMainLayer, TFBertModel, TFBertPreTrainedModel, TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST, WordpieceTokenizer, load_tf_weights_in_bert
from .models.bert_generation import BertGenerationConfig, BertGenerationDecoder, BertGenerationEncoder, BertGenerationTokenizer, load_tf_weights_in_bert_generation
from .models.bert_japanese import BertJapaneseTokenizer, CharacterTokenizer, MecabTokenizer
from .models.bertweet import BertweetTokenizer
from .models.blenderbot import BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP, BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST, BlenderbotConfig, BlenderbotForConditionalGeneration, BlenderbotModel, BlenderbotTokenizer, TFBlenderbotForConditionalGeneration, TFBlenderbotModel
from .models.blenderbot_small import BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP, BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST, BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration, BlenderbotSmallModel, BlenderbotSmallTokenizer, TFBlenderbotSmallForConditionalGeneration, TFBlenderbotSmallModel
from .models.camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST, CamembertConfig, CamembertForCausalLM, CamembertForMaskedLM, CamembertForMultipleChoice, CamembertForQuestionAnswering, CamembertForSequenceClassification, CamembertForTokenClassification, CamembertModel, CamembertTokenizer, CamembertTokenizerFast, TFCamembertForMaskedLM, TFCamembertForMultipleChoice, TFCamembertForQuestionAnswering, TFCamembertForSequenceClassification, TFCamembertForTokenClassification, TFCamembertModel, TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST
from .models.ctrl import CTRLConfig, CTRLForSequenceClassification, CTRLLMHeadModel, CTRLModel, CTRLPreTrainedModel, CTRLTokenizer, CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRL_PRETRAINED_MODEL_ARCHIVE_LIST, TFCTRLForSequenceClassification, TFCTRLLMHeadModel, TFCTRLModel, TFCTRLPreTrainedModel, TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST
from .models.deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, DebertaConfig, DebertaForSequenceClassification, DebertaModel, DebertaPreTrainedModel, DebertaTokenizer
from .models.distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST, DistilBertConfig, DistilBertForMaskedLM, DistilBertForMultipleChoice, DistilBertForQuestionAnswering, DistilBertForSequenceClassification, DistilBertForTokenClassification, DistilBertModel, DistilBertPreTrainedModel, DistilBertTokenizer, DistilBertTokenizerFast, TFDistilBertForMaskedLM, TFDistilBertForMultipleChoice, TFDistilBertForQuestionAnswering, TFDistilBertForSequenceClassification, TFDistilBertForTokenClassification, TFDistilBertMainLayer, TFDistilBertModel, TFDistilBertPreTrainedModel, TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST
from .models.dpr import DPRConfig, DPRContextEncoder, DPRContextEncoderTokenizer, DPRContextEncoderTokenizerFast, DPRPretrainedContextEncoder, DPRPretrainedQuestionEncoder, DPRPretrainedReader, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRQuestionEncoderTokenizerFast, DPRReader, DPRReaderOutput, DPRReaderTokenizer, DPRReaderTokenizerFast, DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST, DPR_PRETRAINED_CONFIG_ARCHIVE_MAP, DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST, DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST, TFDPRContextEncoder, TFDPRPretrainedContextEncoder, TFDPRPretrainedQuestionEncoder, TFDPRPretrainedReader, TFDPRQuestionEncoder, TFDPRReader, TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST, TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST, TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST
from .models.electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST, ElectraConfig, ElectraForMaskedLM, ElectraForMultipleChoice, ElectraForPreTraining, ElectraForQuestionAnswering, ElectraForSequenceClassification, ElectraForTokenClassification, ElectraModel, ElectraPreTrainedModel, ElectraTokenizer, ElectraTokenizerFast, TFElectraForMaskedLM, TFElectraForMultipleChoice, TFElectraForPreTraining, TFElectraForQuestionAnswering, TFElectraForSequenceClassification, TFElectraForTokenClassification, TFElectraModel, TFElectraPreTrainedModel, TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST, load_tf_weights_in_electra
from .models.encoder_decoder import EncoderDecoderConfig, EncoderDecoderModel
from .models.flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST, FlaubertConfig, FlaubertForMultipleChoice, FlaubertForQuestionAnswering, FlaubertForQuestionAnsweringSimple, FlaubertForSequenceClassification, FlaubertForTokenClassification, FlaubertModel, FlaubertTokenizer, FlaubertWithLMHeadModel, TFFlaubertForMultipleChoice, TFFlaubertForQuestionAnsweringSimple, TFFlaubertForSequenceClassification, TFFlaubertForTokenClassification, TFFlaubertModel, TFFlaubertWithLMHeadModel, TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST
from .models.fsmt import FSMTConfig, FSMTForConditionalGeneration, FSMTModel, FSMTTokenizer, FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, PretrainedFSMTModel
from .models.funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST, FunnelBaseModel, FunnelConfig, FunnelForMaskedLM, FunnelForMultipleChoice, FunnelForPreTraining, FunnelForQuestionAnswering, FunnelForSequenceClassification, FunnelForTokenClassification, FunnelModel, FunnelTokenizer, FunnelTokenizerFast, TFFunnelBaseModel, TFFunnelForMaskedLM, TFFunnelForMultipleChoice, TFFunnelForPreTraining, TFFunnelForQuestionAnswering, TFFunnelForSequenceClassification, TFFunnelForTokenClassification, TFFunnelModel, TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST, load_tf_weights_in_funnel
from .models.gpt2 import GPT2Config, GPT2DoubleHeadsModel, GPT2ForSequenceClassification, GPT2LMHeadModel, GPT2Model, GPT2PreTrainedModel, GPT2Tokenizer, GPT2TokenizerFast, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2_PRETRAINED_MODEL_ARCHIVE_LIST, TFGPT2DoubleHeadsModel, TFGPT2ForSequenceClassification, TFGPT2LMHeadModel, TFGPT2MainLayer, TFGPT2Model, TFGPT2PreTrainedModel, TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST, load_tf_weights_in_gpt2
from .models.herbert import HerbertTokenizer, HerbertTokenizerFast
from .models.layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST, LayoutLMConfig, LayoutLMForMaskedLM, LayoutLMForSequenceClassification, LayoutLMForTokenClassification, LayoutLMModel, LayoutLMTokenizer, LayoutLMTokenizerFast
from .models.led import LEDConfig, LEDForConditionalGeneration, LEDForQuestionAnswering, LEDForSequenceClassification, LEDModel, LEDTokenizer, LEDTokenizerFast, LED_PRETRAINED_CONFIG_ARCHIVE_MAP, LED_PRETRAINED_MODEL_ARCHIVE_LIST, TFLEDForConditionalGeneration, TFLEDModel, TFLEDPreTrainedModel
from .models.longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST, LongformerConfig, LongformerForMaskedLM, LongformerForMultipleChoice, LongformerForQuestionAnswering, LongformerForSequenceClassification, LongformerForTokenClassification, LongformerModel, LongformerSelfAttention, LongformerTokenizer, LongformerTokenizerFast, TFLongformerForMaskedLM, TFLongformerForMultipleChoice, TFLongformerForQuestionAnswering, TFLongformerForSequenceClassification, TFLongformerForTokenClassification, TFLongformerModel, TFLongformerSelfAttention, TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST
from .models.lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig, LxmertEncoder, LxmertForPreTraining, LxmertForQuestionAnswering, LxmertModel, LxmertPreTrainedModel, LxmertTokenizer, LxmertTokenizerFast, LxmertVisualFeatureEncoder, LxmertXLayer, TFLxmertForPreTraining, TFLxmertMainLayer, TFLxmertModel, TFLxmertPreTrainedModel, TFLxmertVisualFeatureEncoder, TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST
from .models.marian import MarianConfig, MarianMTModel, MarianModel, MarianTokenizer, TFMarian, TFMarianMTModel
from .models.mbart import MBartConfig, MBartForConditionalGeneration, MBartForQuestionAnswering, MBartForSequenceClassification, MBartModel, MBartTokenizer, MBartTokenizerFast, TFMBartForConditionalGeneration, TFMBartModel
from .models.mmbt import MMBTConfig, MMBTForClassification, MMBTModel, ModalEmbeddings
from .models.mobilebert import MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST, MobileBertConfig, MobileBertForMaskedLM, MobileBertForMultipleChoice, MobileBertForNextSentencePrediction, MobileBertForPreTraining, MobileBertForQuestionAnswering, MobileBertForSequenceClassification, MobileBertForTokenClassification, MobileBertLayer, MobileBertModel, MobileBertPreTrainedModel, MobileBertTokenizer, MobileBertTokenizerFast, TFMobileBertForMaskedLM, TFMobileBertForMultipleChoice, TFMobileBertForNextSentencePrediction, TFMobileBertForPreTraining, TFMobileBertForQuestionAnswering, TFMobileBertForSequenceClassification, TFMobileBertForTokenClassification, TFMobileBertMainLayer, TFMobileBertModel, TFMobileBertPreTrainedModel, TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST, load_tf_weights_in_mobilebert
from .models.mpnet import MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP, MPNET_PRETRAINED_MODEL_ARCHIVE_LIST, MPNetConfig, MPNetForMaskedLM, MPNetForMultipleChoice, MPNetForQuestionAnswering, MPNetForSequenceClassification, MPNetForTokenClassification, MPNetLayer, MPNetModel, MPNetPreTrainedModel, MPNetTokenizer, MPNetTokenizerFast, TFMPNetForMaskedLM, TFMPNetForMultipleChoice, TFMPNetForQuestionAnswering, TFMPNetForSequenceClassification, TFMPNetForTokenClassification, TFMPNetMainLayer, TFMPNetModel, TFMPNetPreTrainedModel, TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST
from .models.mt5 import MT5Config, MT5EncoderModel, MT5ForConditionalGeneration, MT5Model, MT5Tokenizer, MT5TokenizerFast, TFMT5EncoderModel, TFMT5ForConditionalGeneration, TFMT5Model
from .models.openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST, OpenAIGPTConfig, OpenAIGPTDoubleHeadsModel, OpenAIGPTForSequenceClassification, OpenAIGPTLMHeadModel, OpenAIGPTModel, OpenAIGPTPreTrainedModel, OpenAIGPTTokenizer, OpenAIGPTTokenizerFast, TFOpenAIGPTDoubleHeadsModel, TFOpenAIGPTForSequenceClassification, TFOpenAIGPTLMHeadModel, TFOpenAIGPTMainLayer, TFOpenAIGPTModel, TFOpenAIGPTPreTrainedModel, TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST, load_tf_weights_in_openai_gpt
from .models.pegasus import PegasusConfig, PegasusForConditionalGeneration, PegasusModel, PegasusTokenizer, PegasusTokenizerFast, TFPegasusForConditionalGeneration, TFPegasusModel
from .models.phobert import PhobertTokenizer
from .models.prophetnet import PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST, ProphetNetConfig, ProphetNetDecoder, ProphetNetEncoder, ProphetNetForCausalLM, ProphetNetForConditionalGeneration, ProphetNetModel, ProphetNetPreTrainedModel, ProphetNetTokenizer
from .models.rag import RagConfig, RagModel, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration, RagTokenizer
from .models.reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST, ReformerAttention, ReformerConfig, ReformerForMaskedLM, ReformerForQuestionAnswering, ReformerForSequenceClassification, ReformerLayer, ReformerModel, ReformerModelWithLMHead, ReformerTokenizer, ReformerTokenizerFast
from .models.retribert import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST, RetriBertConfig, RetriBertModel, RetriBertPreTrainedModel, RetriBertTokenizer, RetriBertTokenizerFast
from .models.roberta import FlaxRobertaModel, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, RobertaConfig, RobertaForCausalLM, RobertaForMaskedLM, RobertaForMultipleChoice, RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaForTokenClassification, RobertaModel, RobertaTokenizer, RobertaTokenizerFast, TFRobertaForMaskedLM, TFRobertaForMultipleChoice, TFRobertaForQuestionAnswering, TFRobertaForSequenceClassification, TFRobertaForTokenClassification, TFRobertaMainLayer, TFRobertaModel, TFRobertaPreTrainedModel, TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
from .models.squeezebert import SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST, SqueezeBertConfig, SqueezeBertForMaskedLM, SqueezeBertForMultipleChoice, SqueezeBertForQuestionAnswering, SqueezeBertForSequenceClassification, SqueezeBertForTokenClassification, SqueezeBertModel, SqueezeBertModule, SqueezeBertPreTrainedModel, SqueezeBertTokenizer, SqueezeBertTokenizerFast
from .models.t5 import T5Config, T5EncoderModel, T5ForConditionalGeneration, T5Model, T5PreTrainedModel, T5Tokenizer, T5TokenizerFast, T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5_PRETRAINED_MODEL_ARCHIVE_LIST, TFT5EncoderModel, TFT5ForConditionalGeneration, TFT5Model, TFT5PreTrainedModel, TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST, load_tf_weights_in_t5
from .models.tapas import TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP, TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST, TapasConfig, TapasForMaskedLM, TapasForQuestionAnswering, TapasForSequenceClassification, TapasModel, TapasTokenizer
from .models.transfo_xl import AdaptiveEmbedding, TFAdaptiveEmbedding, TFTransfoXLForSequenceClassification, TFTransfoXLLMHeadModel, TFTransfoXLMainLayer, TFTransfoXLModel, TFTransfoXLPreTrainedModel, TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST, TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST, TransfoXLConfig, TransfoXLCorpus, TransfoXLForSequenceClassification, TransfoXLLMHeadModel, TransfoXLModel, TransfoXLPreTrainedModel, TransfoXLTokenizer, load_tf_weights_in_transfo_xl
from .models.xlm import TFXLMForMultipleChoice, TFXLMForQuestionAnsweringSimple, TFXLMForSequenceClassification, TFXLMForTokenClassification, TFXLMMainLayer, TFXLMModel, TFXLMPreTrainedModel, TFXLMWithLMHeadModel, TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST, XLMConfig, XLMForMultipleChoice, XLMForQuestionAnswering, XLMForQuestionAnsweringSimple, XLMForSequenceClassification, XLMForTokenClassification, XLMModel, XLMPreTrainedModel, XLMTokenizer, XLMWithLMHeadModel, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLM_PRETRAINED_MODEL_ARCHIVE_LIST
from .models.xlm_prophetnet import XLMProphetNetConfig, XLMProphetNetDecoder, XLMProphetNetEncoder, XLMProphetNetForCausalLM, XLMProphetNetForConditionalGeneration, XLMProphetNetModel, XLMProphetNetTokenizer, XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST
from .models.xlm_roberta import TFXLMRobertaForMaskedLM, TFXLMRobertaForMultipleChoice, TFXLMRobertaForQuestionAnswering, TFXLMRobertaForSequenceClassification, TFXLMRobertaForTokenClassification, TFXLMRobertaModel, TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, XLMRobertaConfig, XLMRobertaForCausalLM, XLMRobertaForMaskedLM, XLMRobertaForMultipleChoice, XLMRobertaForQuestionAnswering, XLMRobertaForSequenceClassification, XLMRobertaForTokenClassification, XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaTokenizerFast, XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
from .models.xlnet import TFXLNetForMultipleChoice, TFXLNetForQuestionAnsweringSimple, TFXLNetForSequenceClassification, TFXLNetForTokenClassification, TFXLNetLMHeadModel, TFXLNetMainLayer, TFXLNetModel, TFXLNetPreTrainedModel, TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNET_PRETRAINED_MODEL_ARCHIVE_LIST, XLNetConfig, XLNetForMultipleChoice, XLNetForQuestionAnswering, XLNetForQuestionAnsweringSimple, XLNetForSequenceClassification, XLNetForTokenClassification, XLNetLMHeadModel, XLNetModel, XLNetPreTrainedModel, XLNetTokenizer, XLNetTokenizerFast, load_tf_weights_in_xlnet
from .pipelines import Conversation, ConversationalPipeline, CsvPipelineDataFormat, FeatureExtractionPipeline, FillMaskPipeline, JsonPipelineDataFormat, NerPipeline, PipedPipelineDataFormat, Pipeline, PipelineDataFormat, QuestionAnsweringPipeline, SummarizationPipeline, TableQuestionAnsweringPipeline, Text2TextGenerationPipeline, TextClassificationPipeline, TextGenerationPipeline, TokenClassificationPipeline, TranslationPipeline, ZeroShotClassificationPipeline, pipeline
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import AddedToken, BatchEncoding, CharSpan, PreTrainedTokenizerBase, SpecialTokensMixin, TensorType, TokenSpan
from .trainer_callback import DefaultFlowCallback, EarlyStoppingCallback, PrinterCallback, ProgressCallback, TrainerCallback, TrainerControl, TrainerState
from .trainer_utils import EvalPrediction, EvaluationStrategy, SchedulerType, set_seed
from .training_args import TrainingArguments
from .training_args_seq2seq import Seq2SeqTrainingArguments
from .training_args_tf import TFTrainingArguments
from .models.barthez import BarthezTokenizer, BarthezTokenizerFast
from .utils.dummy_sentencepiece_objects import *
from .tokenization_utils_fast import PreTrainedTokenizerFast
from .utils.dummy_tokenizers_objects import *
from .benchmark.benchmark import PyTorchBenchmark
from .benchmark.benchmark_args import PyTorchBenchmarkArguments
from .data.data_collator import DataCollator, DataCollatorForLanguageModeling, DataCollatorForPermutationLanguageModeling, DataCollatorForSOP, DataCollatorForTokenClassification, DataCollatorForWholeWordMask, DataCollatorWithPadding, default_data_collator
from .data.datasets import GlueDataTrainingArguments, GlueDataset, LineByLineTextDataset, LineByLineWithRefDataset, LineByLineWithSOPTextDataset, SquadDataTrainingArguments, SquadDataset, TextDataset, TextDatasetForNextSentencePrediction
from .generation_beam_search import BeamScorer, BeamSearchScorer
from .generation_logits_process import HammingDiversityLogitsProcessor, LogitsProcessor, LogitsProcessorList, LogitsWarper, MinLengthLogitsProcessor, NoBadWordsLogitsProcessor, NoRepeatNGramLogitsProcessor, PrefixConstrainedLogitsProcessor, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from .generation_utils import top_k_top_p_filtering
from .modeling_utils import Conv1D, PreTrainedModel, apply_chunking_to_forward, prune_layer
from .optimization import Adafactor, AdamW, get_constant_schedule, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_scheduler
from .trainer import Trainer
from .trainer_pt_utils import torch_distributed_zero_first
from .trainer_seq2seq import Seq2SeqTrainer
from .utils.dummy_pt_objects import *
from .benchmark.benchmark_args_tf import TensorFlowBenchmarkArguments
from .benchmark.benchmark_tf import TensorFlowBenchmark
from .generation_tf_utils import tf_top_k_top_p_filtering
from .modeling_tf_utils import TFPreTrainedModel, TFSequenceSummary, TFSharedEmbeddings, shape_list
from .optimization_tf import AdamWeightDecay, GradientAccumulator, WarmUp, create_optimizer
from .trainer_tf import TFTrainer
from .utils.dummy_tf_objects import *
from .modeling_flax_utils import FlaxPreTrainedModel
from .utils.dummy_flax_objects import *

__version__ = "4.2.1"
logger = logging.get_logger(__name__)
_import_structure = { "configuration_utils": ["PretrainedConfig"],"data": ["DataProcessor", "InputExample", "InputFeatures", "SingleSentenceClassificationProcessor", "SquadExample", "SquadFeatures", "SquadV1Processor", "SquadV2Processor", "glue_compute_metrics", "glue_convert_examples_to_features", "glue_output_modes", "glue_processors", "glue_tasks_num_labels", "squad_convert_examples_to_features", "xnli_compute_metrics", "xnli_output_modes", "xnli_processors", "xnli_tasks_num_labels"],"file_utils": ["CONFIG_NAME", "MODEL_CARD_NAME", "PYTORCH_PRETRAINED_BERT_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "SPIECE_UNDERLINE", "TF2_WEIGHTS_NAME", "TF_WEIGHTS_NAME", "TRANSFORMERS_CACHE", "WEIGHTS_NAME", "add_end_docstrings", "add_start_docstrings", "cached_path", "is_apex_available", "is_datasets_available", "is_faiss_available", "is_flax_available", "is_psutil_available", "is_py3nvml_available", "is_sentencepiece_available", "is_sklearn_available", "is_tf_available", "is_tokenizers_available", "is_torch_available", "is_torch_tpu_available"],"hf_argparser": ["HfArgumentParser"],"integrations": ["is_comet_available", "is_optuna_available", "is_ray_available", "is_ray_tune_available", "is_tensorboard_available", "is_wandb_available"],"modelcard": ["ModelCard"],"modeling_tf_pytorch_utils": ["convert_tf_weight_name_to_pt_weight_name", "load_pytorch_checkpoint_in_tf2_model", "load_pytorch_model_in_tf2_model", "load_pytorch_weights_in_tf2_model", "load_tf2_checkpoint_in_pytorch_model", "load_tf2_model_in_pytorch_model", "load_tf2_weights_in_pytorch_model"],"models": [],"models.albert": ["ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "AlbertConfig"],"models.auto": ["ALL_PRETRAINED_CONFIG_ARCHIVE_MAP", "CONFIG_MAPPING", "MODEL_NAMES_MAPPING", "TOKENIZER_MAPPING", "AutoConfig", "AutoTokenizer"],"models.bart": ["BartConfig", "BartTokenizer"],"models.barthez": [],"models.bert": ["BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BasicTokenizer", "BertConfig", "BertTokenizer", "WordpieceTokenizer"],"models.bert_generation": ["BertGenerationConfig"],"models.bert_japanese": ["BertJapaneseTokenizer", "CharacterTokenizer", "MecabTokenizer"],"models.bertweet": ["BertweetTokenizer"],"models.blenderbot": ["BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BlenderbotConfig", "BlenderbotTokenizer"],"models.blenderbot_small": ["BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP", "BlenderbotSmallConfig", "BlenderbotSmallTokenizer"],"models.camembert": ["CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CamembertConfig"],"models.ctrl": ["CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP", "CTRLConfig", "CTRLTokenizer"],"models.deberta": ["DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "DebertaConfig", "DebertaTokenizer"],"models.distilbert": ["DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DistilBertConfig", "DistilBertTokenizer"],"models.dpr": ["DPR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DPRConfig", "DPRContextEncoderTokenizer", "DPRQuestionEncoderTokenizer", "DPRReaderOutput", "DPRReaderTokenizer"],"models.electra": ["ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP", "ElectraConfig", "ElectraTokenizer"],"models.encoder_decoder": ["EncoderDecoderConfig"],"models.flaubert": ["FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FlaubertConfig", "FlaubertTokenizer"],"models.fsmt": ["FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FSMTConfig", "FSMTTokenizer"],"models.funnel": ["FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP", "FunnelConfig", "FunnelTokenizer"],"models.gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2Tokenizer"],"models.herbert": ["HerbertTokenizer"],"models.layoutlm": ["LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "LayoutLMConfig", "LayoutLMTokenizer"],"models.led": ["LED_PRETRAINED_CONFIG_ARCHIVE_MAP", "LEDConfig", "LEDTokenizer"],"models.longformer": ["LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "LongformerConfig", "LongformerTokenizer"],"models.lxmert": ["LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LxmertConfig", "LxmertTokenizer"],"models.marian": ["MarianConfig"],"models.mbart": ["MBartConfig"],"models.mmbt": ["MMBTConfig"],"models.mobilebert": ["MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MobileBertConfig", "MobileBertTokenizer"],"models.mpnet": ["MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "MPNetConfig", "MPNetTokenizer"],"models.mt5": ["MT5Config"],"models.openai": ["OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "OpenAIGPTConfig", "OpenAIGPTTokenizer"],"models.pegasus": ["PegasusConfig"],"models.phobert": ["PhobertTokenizer"],"models.prophetnet": ["PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "ProphetNetConfig", "ProphetNetTokenizer"],"models.rag": ["RagConfig", "RagRetriever", "RagTokenizer"],"models.reformer": ["REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "ReformerConfig"],"models.retribert": ["RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RetriBertConfig", "RetriBertTokenizer"],"models.roberta": ["ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "RobertaConfig", "RobertaTokenizer"],"models.squeezebert": ["SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "SqueezeBertConfig", "SqueezeBertTokenizer"],"models.t5": ["T5_PRETRAINED_CONFIG_ARCHIVE_MAP", "T5Config"],"models.tapas": ["TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP", "TapasConfig", "TapasTokenizer"],"models.transfo_xl": ["TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP", "TransfoXLConfig", "TransfoXLCorpus", "TransfoXLTokenizer"],"models.xlm": ["XLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMConfig", "XLMTokenizer"],"models.xlm_prophetnet": ["XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMProphetNetConfig"],"models.xlm_roberta": ["XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMRobertaConfig"],"models.xlnet": ["XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLNetConfig"],"pipelines": ["Conversation", "ConversationalPipeline", "CsvPipelineDataFormat", "FeatureExtractionPipeline", "FillMaskPipeline", "JsonPipelineDataFormat", "NerPipeline", "PipedPipelineDataFormat", "Pipeline", "PipelineDataFormat", "QuestionAnsweringPipeline", "SummarizationPipeline", "TableQuestionAnsweringPipeline", "Text2TextGenerationPipeline", "TextClassificationPipeline", "TextGenerationPipeline", "TokenClassificationPipeline", "TranslationPipeline", "ZeroShotClassificationPipeline", "pipeline"],"tokenization_utils": ["PreTrainedTokenizer"],"tokenization_utils_base": ["AddedToken", "BatchEncoding", "CharSpan", "PreTrainedTokenizerBase", "SpecialTokensMixin", "TensorType", "TokenSpan"],"trainer_callback": ["DefaultFlowCallback", "EarlyStoppingCallback", "PrinterCallback", "ProgressCallback", "TrainerCallback", "TrainerControl", "TrainerState"],"trainer_utils": ["EvalPrediction", "EvaluationStrategy", "SchedulerType", "set_seed"],"training_args": ["TrainingArguments"],"training_args_seq2seq": ["Seq2SeqTrainingArguments"],"training_args_tf": ["TFTrainingArguments"],"utils": ["logging"] }
if is_sentencepiece_available():
    ...
else:
    ...
if is_tokenizers_available():
    ...
else:
    ...
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
if is_sentencepiece_available():
    ...
else:
    ...
if is_tokenizers_available():
    ...
else:
    ...
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
if not is_tf_available() and not is_torch_available() and not is_flax_available():
    ...
