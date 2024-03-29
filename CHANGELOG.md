# v0.2.0 (2021-01-22)

# Refactor

## Data

- **data**: log error when packaged scripts are set incorrectly
- **data**: specify script version via an argument
- **data**: move base converter setup into base class
- **data**: rename Seq2SeqBinarizer -> TranslationBinarizer
- **data**: move default binarizer to its own file
- **data**: rename Seq2SeqDataset -> TranslationDataset
- **tokenizers**: move model_max_length to tokenizer base class
- **data**: rename DataCollatorForSeq2Seq -> DataCollatorForTranslation
- **code**: parametrize code pre-tokenization arguments
- **code**: add code specific additional tokens to code tokenizer
- **code**: add code special tokens to additional tokens list
- **data**: simplify torch.full passed arguments to avoid facing unsupported behavior for pytorch < 1.7
- **data**: rename DataCollatorForSeq2SeqDenoising -> DataCollatorForWholeWordMasking
- **data**: add base abstract classes for data collator API
- **tokenizers**: add model_max_length property signature
- **data**: truncate input to tokenizer associated model_max_length
- **data**: match features keys to huggingface models signature
- **data**: rename dataset output keys to match huggingface models signature
- **tokenizers**: remove outdated bug workaround for saving legacy tokenizers with AddedToken tokens
- **tokenizers**: avoid save_directory from adding into tokenizer assets
- **tokenizers**: save tokenizers in a new unified format by default
- **data**: move common binarization logic to BinarizerBase class
- **data**: rename TransformerDatasetConverter -> DefaultDatasetConverter
- **data**: rename TransformerBinarizer -> DefaultBinarizer
- **tokenizers**: move roberta tokenizer trainer logic to gpt2 tokenizer trainer

## Common

- **common**: rename PartialInitable -> FromPartial
- **common**: raise runtime error when no registry found
- **common**: raise runtime error when no value found
- **common**: explicitly tell class name in error
- **common**: move generic-based registry to its own container class

## Modules

- **modules**: remove deprecated seq2seq neural modules
- **modules**: leave only basic setup in TransformerModule class
- **modules**: make TransformerDataModule an abstract base class for specific data modules
- **modules**: move loss calculation to a helper method
- **modules**: rewrite label smoothing loss calculation wrt ignored indices
- **modules**: return only loss in training_step method
- **modules**: do not move valid perplexity value to cpu
- **modules**: do not move perplexity value to cpu
- **modules**: remove unused metric buffers
- **callbacks**: use callback properties for storing monitor values and formatting checkpoints
- **modules**: calculate the perplexity based on already computed loss value
- **modules**: move getting the learning rate to an instance property
- **modules**: remove deprecated prepare_batch step for making batches flatten
- **modules**: specify data iterator's collate function for a dataloader
- **modules**: use fast tokenizer as the default
- **modules**: use DatasetIterator types for dataset iterators
- **metrics**: use a computed loss value for calculating the perplexity
- **modules**: rename TokenizerModule -> TokenizerTrainer, remove non-training proxy methods

## Tasks

- **tasks**: use Denoising modules for DenoisingTask & use default params
- **tasks**: use MaskedLM modules for MaskedLMTask
- **code**: rename code-roberta -> code_roberta
- **tasks**: rename Seq2SeqTask -> DenoisingTask
- **tasks**: rename TransformerTask -> MaskedLMTask
- **tasks**: pass positional embeddings model config args
- **tasks**: move tokenizer trainer base to data/tokenizers dir
- **tasks**: move roberta tokenizer trainer to data/tokenizers dir
- **tasks**: move roberta tokenizer to data/tokenizers dir
- **tasks**: move tokenizer base to data/tokenizers dir
- **tasks**: register roberta tokenizer trainer in TokenizerTrainer class
- **tasks**: use renamed TokenizerTrainerBase class for tokenizer trainer
- **tasks**: rename class BaseTokenizerTrainer -> TokenizerTrainerBase
- **code**: move tokenizer training from CodeBBPETokenizerModule -> CodeRobertaTokenizerTrainer
- **code**: rename code_dataset_converter -> dataset_converter_code
- **code**: rename CodeBBPETokenizerFast -> CodeRobertaTokenizer
- **tasks**: move tokenizer training from ByteLevelBPETokenizerModule -> RobertaTokenizerTrainer
- **tasks**: mirror RobertaTokenizer instead of ByteLevelBPETokenizerFast

## Training

- **training**: update tokenizer params items with non-existing keys from kwargs
- **training**: update model config items with non-existing keys from kwargs
- **training**: load tokenizer from config with flatten components
- **training**: load model from config with flatten components
- **training**: use fast tokenizer as the default

## CLI

- **cli**: save tokenizer after preprocessing as some updated might happen during the preprocessing
- **cli**: load tokenizer from pretrained fast tokenizer class

## Other

- **contrib**: move lightning contrib types to lightning dir
- **utils**: make recursive subclass finding more expressive

# Features

## Data

- **data**: add offline mode support by passing none to script version
- **tokenizers**: add additional tokens to vocab after adding trained vocab
- **tokenizers**: support additional tokens added after training on subword merges
- **data**: add data collator for bart denoising unsupervised objective
- **tokenizers**: set model_max_length explicitly for new tokenizers while training
- **data**: use tokenizer model associated input max length if no expicit value is set
- **data**: add dataset iterator built-in collate function for making batches flatten
- **data**: add a binarizer for se2seq parallel datasets processing
- **data**: handle empty sequences edge-case
- **tokenizers**: add **call** method to tokenizer protocol
- **data**: add seq2seq collators for training denoising autoencoder models
- **data**: add a seq2seq dataset for s2s tasks
- **tokenizers**: add BART tokenizer trainer inherited from roberta tokenizer trainer
- **tokenizers**: add BART seq2seq registrable tokenizer
- **tokenizers**: add GPT2 registrable tokenizer
- **tokenizers**: add shared tokenizer training parameters
- **tokenizers**: add seq2seq trainer interface + some basic types

## Common

- **common**: use a magic MISSING value for building mergable dataclasses

## Modules

- **modules**: add WordLM modules for whole word masking based on translation modules
- **modules**: add lightning modules for training denoising models aka BART
- **modules**: add Translation lightning modules for training mono/bilingual translation models
- **modules**: add MaskedLM lightning modules for training bert-like models
- **modules**: support translation data collator for nmt tasks
- **modules**: choose data collator via datamodule config
- **modules**: log step metric implicitly to ensure that we always start with the latest global step
- **modules**: log global step to track the common training state
- **modules**: add a seq2seq module for training seq2seq models
- **modules**: add a seq2seq datamodule class for seq2seq data setup
- **modules**: add a label smoothing criterion module

## Tasks

- **tasks**: add WordLM task for training whole word lm models
- **tasks**: add translation task for training translation models
- **tasks**: define base task params dataclass
- **tasks**: parametrize vocab size with added tokens
- **tasks**: add a seq2seq task for training seq2seq models
- **tasks**: add a method to find params of a certain type
- **code**: make code roberta tokenizer registrable
- **tasks**: make roberta tokenizer a registrable component
- **tasks**: add base registrable class for tokenizers
- **tasks**: define special token ids for roberta tokenizer
- **code**: register code dataset converter
- **tasks**: add base tokenizer trainer class with defined method signatures

## CLI

- **cli**: specify tokenizer component in user args
- **cli**: register default subcommands prior to registering user-defined subcomands

## Other

- **contrib**: add a script to parse jsonlines datasets
- **utils**: add a method to update dict with non-existing keys inplace

# Fix

- **data**: make sure sequences are converted to tensors
- **code**: return workaround to avoid newline char disambiguation
- **callbacks**: get all logged scalar metrics
- **metrics**: avoid persisting module state in state_dict
- **data**: avoid masking special tokens
- **data**: use an up-to-date nonzero method override
- **data**: check if padding required for ascending sequences
- **data**: make back-encoding idemponent operation
- **training**: enable deterministic mode if seed is set
- **tasks**: use roberta post-processing for roberta tokenization pipeline
- **tasks**: save pretrained tokenizer with respect to legacy mode

# v0.1.14 (2020-10-30)

# Refactor

# Tasks

- **tasks**: remove unused generic type
- **code**: remove unused `dropout` in code tokenizer fast
- **tasks**: rename `tokenizer_path` -> `save_directory`
- **code**: override template methods to use code bbpe fast tokenizer
- **tasks**: rename `FastTokenizer` -> `BaseTokenizer`
- **tasks**: get path strings via direct conversion to str object
- **tasks**: conform to base tokenizer module class signatures
- **tasks**: simplify making tokenizer_path a pathlike object
- **tasks**: save whole pre-trained tokenizer instead of just a model
- **tasks**: rename `fix_tokenizer` -> `__fix_tokenizer`
- **tasks**: remove dropout arg from base tokenizer fast

# CLI

- **cli**: use command params for getting pretrained tokenizer path
- **cli**: remove file paths from tokenizer configs while saving
- **cli**: remove `parse_repository` empty file

# Features

- **cli**: support tokenizer saving with `legacy_format` flag
- **tasks**: add `tokenizer_file` in pretrained data configs
- **tasks**: inject fast tokenizer user-defined args while configuring one
- **tasks**: use pre-trained rust tokenizer while configuring
- **tasks**: ignore training args while saving pre-trained tokenizer
- **tasks**: define main methods signature in `TransformerTokenizerModule`
- **tasks**: setup fast tokenizer pipeline with tokenizers lib components

# Fix

- **tasks**: do not inject params values into pre-trained tokenizer

# v0.1.13 (2020-10-27)

# Refactor

- **tasks**: export code bbpe tokenization components
- **code**: use code bbpe tokenizer in code bbpe tokenization module
- **code**: inherit code bbpe tokenization from byte level bpe tokenizer
- **tasks**: use byte level bpe tokenizer in bbpe tokenization module
- **tasks**: move base tokenization interface to TransformerTokenizerModule
- **tasks**: inherit bbpe fast tokenizer from roberta tokenizer
- **cli**: rewrite init_kwargs through dynamic attributes
- **tasks**: rewrite init_kwargs through dynamic attributes
- add configured logger to main files
- **training**: add configured logger for training

# Fix

- **tasks**: create parent directory if one doesn't exist yet

# Features

- **typings**: add transformers auto-generated stubs for pylance support

# v0.1.12 (2020-10-22)

# Refactor

- **modules**: add configured loggers
- **data**: remove super slow mmap file warmup

# Fix

- **modules**: detach output tensors to avoid double backward pass
- **samplers**: remove sampling more than max_tokens in a batch
- **callbacks**: create parent dirs if they do not exist

# v0.1.11 (2020-10-20)

# Refactor

- **callbacks**: update method args for lightning 1.0
- **callbacks**: save checkpoint on training epoch end
- **modules**: prepare clear tensors for logging
- **training**: specify trainer type for better ide help

# Fix

- **cli**: import an existing module attribute

# v0.1.9 (2020-10-16)

# v0.1.8 (2020-10-16)

# Features

- **code**: add tokenizer module for source code nlp tasks
- **code**: add fast tokenizer for source code nlp tasks
- **code**: add dataset converter for source code nlp tasks
- **data**: add split_chunks parameter to control the writing strategy
- **data**: add padding parameter for padding strategy selection

# Refactor

- **cli**: update names for imported classes
- **data**: rename FlatBinarizer -> TransformerBinarizer
- **data**: remove code preprocessing from common component
- **tasks**: optimize parsing pretrained args from params
- **tasks**: remove code special tokens from base classes

# v0.1.7 (2020-10-15)

# Refactor

- **cli**: use correct paths to import modules
- **tasks**: add transformer task class
- **base_transformers**: remove moved files
- **base_transformers**: move transformer trainer to training dir
- **base_transformers**: move transformer tokenizer module to tasks dir
- **base_transformers**: move tokenization to tasks dir
- **base_transformers**: move task module to tasks dir
- **base_transformers**: move tokenizer module to modules dir
- **base_transformers**: move module and datamodule to modules dir
- **base_transformers**: move loading from config to training dir
- **base_transformers**: move metrics to modules/metrics
- **base_transformers**: move callbacks to modules/callbacks
- **tasks**: remove codebert deprecated task
- **data**: shut the static checker for numpy data types

# Features

- **contrib**: add lightning trainer protocol with properties

# v0.1.5 (2020-10-12)

# Refactor

- **changelog**: rename gitnetic -> formerbox
- rename gitnetic -> formerbox

# v0.1.0 (2020-10-11)

# Fix

# CLI

- **cli**: keep only token-related items in the tokenizer config
- **cli**: duplicate args to avoid errors in subparsers
- **cli**: save pretrained tokenizer
- **cli**: parse and inject dynamic args in a callback

# Base Transformers

- **base_transformers**: remove training args from token config
- **base_transformers**: skip watching models as it causes pickle errors
- **base_transformers**: update & sync metrics with logging connector
- **base_transformers**: make dir for checkpoints on fit begin
- **base_transformers**: check imported tokenizer class correctly
- **base_transformers**: compare torch tensors with torch comparator
- **base_transformers**: check if imported type is expected
- **base_transformers**: check union types properly
- **base_transformers**: use updated filename to import correctly
- **base_transformers**: use num_gpus numeric value instead of a function
- **base_transformers**: clear extra files after saving the new ones
- **base_transformers**: do not remove reserved checkpoints
- **base_transformers**: add missing module import
- **base_transformers**: return input batch on transfer_batch_to_device

# Common

- **common**: remove property decorator for class attribute
- **common**: make ParamsType non covariant type
- **common**: make entry geenric less strict
- **common**: set default non-missing value for list fields
- **common**: iterate over all params items
- **common**: return entry type from registrable decorator
- **common**: remove unconstrained generics assert check

# Data

- **data**: avoid splitting errors due to formats diff
- **data**: use master branch script for loading text datasets
- **data**: workaround to avoid disambiguation in parsing text datasets
- **data**: add a workaround for super slow long docs tokenization
- **data**: prepare the output dir for results
- **data**: ensure the filepath is a string path
- **data**: skip saving empty instances after pretokenization
- **data**: return a copy of writable data to prevent unsafe behavior
- **data**: close numpy memmap with an underlying instance
- **data**: use the correct length from dim_offsets
- **data**: return the original file offset after reading a buffer
- **data**: use prod operation to get the total size of buffer across all dims
- **data**: export samplers from samplers module

# Utils

- **utils**: unwrap optional loss value properly
- **utils**: tokenize string literals with a prefix without an extra space between
- **utils**: move replacement below as in the original impl

# Codebert

- **codebert**: write special token state dict instead of a string value
- **codebert**: pass required positional arguments
- **codebert**: import batch samplers from samplers module
- **codebert**: use relative import for class within the same module
- **codebert**: add a workaround that prevents from saving pretrained tokenizer

# Optim

- **optim**: align polynomial decay scheduler with original bert paper impl

# Misc

- **changelog**: use correct relative import from commitizen module

# Refactor

# CLI

- **cli**: check if setup params exist
- **cli**: check if parsed params exist
- **cli**: use renamed methos for tokenizer module setup
- **cli**: use updated methods to setup datasets processors
- **cli**: use updated methods for settung up the task
- **cli**: remove extra arg for number of processes
- **cli**: use static params for train subcommand
- **cli**: use static params for convert_dataset subcommand
- **cli**: use an updated binarizer api
- **cli**: impl subcommand for dataset preprocessing
- **cli**: impl subcommand for training a tokenizer
- **cli**: use module logger for printing infos
- **cli**: pass a description help for convert_dataset command
- **cli**: remove main cli method invoke in a subcommand
- **cli**: make convert_dataset a cli subcommand
- **cli**: add explicit tqdm callback function
- **cli**: remove pretokenize command replaced by convert dataset
- **cli**: use the latest parsed params
- **cli**: use a modular binarizer for preprocessing text inputs
- **cli**: use modular tokenizer module for training a new tokenizer
- **cli**: use modular tokenizer module for preprocessing
- **cli**: use modular tokenizer and tokenizer trainer for training
- **cli**: use modular tokenizer class for dataset preprocessing
- **cli**: update import path
- **cli**: use argparse named args to build a trainer
- **cli**: pass index_filepath to builder init
- **cli**: use dataset setup class instead of custom methods to init a dataset
- **cli**: remove unnecessary step that adds special tokens

# Base Transformers

- **base_transformers**: calc perplexity with a new metric class
- **base_transformers**: use a dataclass dict output for transformer models
- **base_transformers**: call super init method
- **base_transformers**: specify params type explicitly
- **base_transformers**: check if parameters fit into integer type
- **base_transformers**: use lightning trainer with typed params
- **base_transformers**: return arbitrary dicts in step methods
- **base_transformers**: remove hparams save as we save the params instance
- **base_transformers**: check if params exist
- **base_transformers**: make tokenizer module generic
- **base_transformers**: use specified save directory or take one from params
- **base_transformers**: use a model output dataclass instead of a plain dict
- **base_transformers**: use updated AdamW optimizer
- **base_transformers**: adjust base task module to HasParableParams protocol
- **base_transformers**: conform to HasParsableParams protocol
- **base_transformers**: rename constructor name from_args -> from_partial
- **base_transformers**: adjust base class for HasParsableParams protocol
- **base_transformers**: impl HasParsableParams protocol
- **base_transformers**: use a composition of protocols to impl static dataclass params
- **base_transformers**: use generics through protocol
- **base_transformers**: use static typed dataclasses for tokenizer module args
- **base_transformers**: impl argument registrable for tokenizer modules
- **base_transformers**: remove outdated training setup actions
- **base_transformers**: use static typed dataclass params in transformer trainer
- **base_transformers**: use static typed dataclass params for task setup
- **base_transformers**: use static typed params with dataclasses in modules
- **base_transformers**: get rid of training params in base class
- **base_transformers**: use generic type bound on the task type
- **base_transformers**: use renamed dataset base class
- **base_transformers**: rename data_iterators -> dataset_iterators
- **base_transformers**: reuse modular task components
- **base_transformers**: add a modular base class for tasks
- **base_transformers**: return result objects on train/valid steps
- **base_transformers**: impl updated API for transformer tokenizer module
- **base_transformers**: remove dynamic setup from transformer fast tokenizer
- **base_transformers**: merge trainer and tokenizer module together
- **base_transformers**: import base classes for tokenizer and trainer
- **base_transformers**: wrap all named args to correctly handle inputs
- **base_transformers**: inherit from base tokenizer trainer
- **base_transformers**: make add_argparse_args abstract in base class
- **base_transformers**: make save_pretrained abstract in base class
- **base_transformers**: move tokenizer trainer to own file
- **base_transformers**: use transformer tokenizer instead of roberta
- **base_transformers**: rename SpecialToken -> Token
- **base_transformers**: use kwargs for building objects
- **base_transformers**: import FromArgs from a common module
- **base_transformers**: move FromArgs to common module
- **base_transformers**: rename InitFromArgsMixin -> FromArgs
- **base_transformers**: make default values to args in initializer
- **base_transformers**: rename params -> config_params
- **base_transformers**: make trainer setup much easier with tasks
- **base_transformers**: inherit init from args behavior for module classes
- **base_transformers**: add init from args extension to TrainingParams
- **base_transformers**: use a more specific tokenizer type for hints
- **base_transformers**: use injected data collator other than hardcoded
- **base_transformers**: rename base_lm -> base_modules
- **base_transformers**: inherit directly from LightningModule
- **base_transformers**: make BaseTrainingMixin lightning agnostic
- **base_transformers**: remove max_steps force override
- **base_transformers**: load dataset iterators in module setup
- **base_transformers**: rename per_gpu_samples -> per_device_samples
- **base_transformers**: use batch_nums from dataset iters for total steps calc
- **base_transformers**: remove redundant file deletion
- **base_transformers**: parametrize the num of checkpoints to keep
- **base_transformers**: rework training script with functional components
- **base_transformers**: rename BaseLMDataModule -> TransformerDataModule
- **base_transformers**: rename BaseDataModuleMixin -> DataLoadingMixin
- **base_transformers**: rename BaseLMTransformer -> TransformerModule
- **base_transformers**: make trainer buildable from functional components
- **base_transformers**: remove data setup from a transformer module
- **base_transformers**: remove unused properties
- **base_transformers**: add back-support for dataloaders with batch samplers
- **base_transformers**: remove deprecated properties and classes
- **base_transformers**: remove reduntand dataset_impl arg
- **base_transformers**: use an inference method for reading datasets
- **base_transformers**: use a simplified method to get a dataloader
- **base_transformers**: dataset iterator implements making a batch sampler, so remove it
- **base_transformers**: pass a datamodule with prepared datasets
- **base_transformers**: use max_steps instead of epochs
- **base_transformers**: pass dataset_impl arg to build a dataset
- **base_transformers**: use max_tokens and batch_size from args
- **base_transformers**: build dataset from the setup type
- **base_transformers**: use a base indexed dataset class for arg types
- **base_transformers**: use updated UniformBatchSampler sampler
- **base_transformers**: import IndexedDataset from data module
- **base_transformers**: impot sampler classes from samplers module
- **base_transformers**: move config parsing to a separate method
- **base_transformers**: add type hint for dataloaders
- **base_transformers**: move trainer to instance attributes scope
- **base_transformers**: calm down unused-argument warning
- **base_transformers**: move class attributes to instance

# Common

- **common**: check if module_finder is FileFinder
- **common**: remove redundant ArgumentRegistrable class
- **common**: inherit from renamed PartialInitable class
- **common**: rename FromArgs -> PartialInitable
- **common**: avoid creation of a new parser in adding parsing args
- **common**: make dataclass_types an instance property
- **common**: fix typing hints in parse method
- **common**: use internal attributes property for keys
- **common**: build instances from named args
- **common**: return a tuple of class and init method
- **common**: remove unused imports
- **common**: make sure cls objects are callable and return inferred T instances

# Data

- **data**: make base binarizer class generic
- **data**: make dataset converter generic
- **data**: remove typechecked decorator
- **data**: move append_path_suffix method to utils
- **data**: remove unused params field
- **data**: conform binarizer to HasParsableParams protocol
- **data**: rename constuctor name from_args -> from_partial
- **data**: conform dataset converter to HasParsableParams protocol
- **data**: rename constructor name from_args -> from_partial
- **data**: conform dataset setup to HasParsableParams protocol
- **data**: define static params in daraset converter base class
- **data**: use 🤗datasets for loading and binarizing text datasets
- **data**: ensure the length value exists
- **data**: make prefetch method abstract
- **data**: move magic code to a protocol type
- **data**: use static typed dataclass args for binarizers
- **data**: use static typed dataclass args in dataset setup
- **data**: print raised errors with logger warnings
- **data**: add static typed argparse args without creating a new parser
- **data**: add typeguard runtime checks
- **data**: use renamed indexed dataset base class
- **data**: use renamed base class
- **data**: rename MMapIndexedDatasetMixin -> MMapIndexedDatasetBase
- **data**: rename IndexedDatasetBuilderMixin -> IndexedDatasetBuilderBase
- **data**: rename IndexedDatasetMixin -> IndexedDatasetBase
- **data**: rename data_iterators -> dataset_iterators
- **data**: make binarizer class modular, move all impl to flat-binarizer
- **data**: catch and log tokenization errors
- **data**: rename tokenizer_max_length -> max_length
- **data**: make underlying sampled index batches property clear
- **data**: fix cycling imports
- **data**: use dataset setup class for binarization init
- **data**: make dataset builder dynamically on binarization
- **data**: remove unused imports
- **data**: build a dataset instance from the given class
- **data**: close the dataset builder open stream on delete
- **data**: move stream to the base class and open on init
- **data**: prepare data and index filepaths in the base class
- **data**: rename pointers -> data_offsets to match a naming convention
- **data**: make magic_code a class instance
- **data**: move dataset builder api to a base mixin class
- **data**: make mmap dataset inheritance through its mixin
- **data**: move len impl to the base class
- **data**: remove context state management
- **data**: close the data stream if one is open
- **data**: remove debug printing
- **data**: make start/end indexing simpler
- **data**: inherit from IndexedDatasetMixin class
- **data**: add a mmap dataset mixin base class with attributes decl
- **data**: remove unused decorator import
- **data**: make indexing simpler
- **data**: use dtype itemsize property instead of element_size
- **data**: add domain specific property to the derived class
- **data**: remove too specific properties and methods from base class
- **data**: use a correct dtype type hint
- **data**: make prefetch a method of a base class for indexed datasets
- **data**: move index validation to its own method
- **data**: move IndexedDataset class attributes and common methods to a mixin base class
- **data**: move finalize() method above
- **data**: return known type ensured by future annotations
- **data**: add **exit** method argument type hints
- **data**: pass index_filepath to init method
- **data**: make sure the stream is always open on write
- **data**: pass index_filepath to init to make finalize() more clear
- **data**: close stream on deinit
- **data**: use numpy fromfile method to read a chunk of bytes
- **data**: move max_token_batch_sampler to samplers dir
- **data**: move distributed_batch_sampler to samplers dir
- **data**: organize imports
- **data**: rename unclear file name f to readable index_file
- **data**: fix dataset builder stream typing
- **data**: ignore numpy binary io
- **data**: remove unnecessary ignore markers
- **data**: wrap int32 type to dtype object
- **data**: fix typing for binary file io

# Utils

- **utils**: return float tensor for perplexity
- **utils**: optimize typing imports
- **utils**: use more_itertools to impl lazy_groups_of method
- **utils**: return generic subclasses
- **utils**: rename special token items to use in tokenizer
- **utils**: tokenize with bleu rules if user sets a flag
- **utils**: return flatten tensors
- **utils**: tokenize spaces with a special token value
- **utils**: rename get_perplexity -> perplexity
- **utils**: use inner clone_repo method from the library
- **utils**: return a path to the cloned repository
- **utils**: use functions instead of classes for commit parsing

# Codebert

- **codebert**: add deprecation warning & fix type checking
- **codebert**: dismiss type checker warnings as the module is deprecated
- **codebert**: use updated UniformBatchSampler sampler type
- **codebert**: import sampler classes from samplers module
- **codebert**: use existing method for assigning weight decays
- **codebert**: use renamed perplexity method
- **codebert**: call empty_cache from module alias
- **codebert**: fix tensor typing without using a cast method
- **codebert**: rename model to roberta_lm
- **codebert**: make helper methods private
- **codebert**: move making params with weight decay to weight_decay_params method

# Optim

# Samplers

- **samplers**: use renamed indexed dataset base class
- **samplers**: inherit from UniformMaxTokensBatchSampler
- **samplers**: cache the number of batches
- **samplers**: rename file with uniform samplers to uniform_batch_sampler
- **samplers**: rename MaxTokensBatchSampler -> UniformBatchSampler
- **samplers**: pass base class args to super initializer
- **samplers**: init parent base sampler with current batch sampler obj
- **samplers**: add batch sampler standard attributes to the base class
- **samplers**: use a base mixin class for data_source

# Misc

- get rid of typeguard typechecked decorator
- **tests**: rename resources -> fixures
- **changelog**: import Commit type from objects module
- pass registrable classes for listing available choices in params
- **changelog**: use a correct title for features
- listen to filelock warnings only
- apply black20 formatting
- **base_components**: use flatten tensors
- remove annotations module for python3.6 compatibility
- **env**: move env variables to .envrc handled by direnv
- rename src -> gitnetic to match the project name

# Features

# CLI

- **cli**: save the pretrained tokenizer to the output directory
- **cli**: import user module from user dir path
- **cli**: preprocess all dataset splits
- **cli**: add functional methods for preprocessing
- **cli**: export train subcommand
- **cli**: add a train cli subcommand
- **cli**: import convert_dataset subcommand
- **cli**: invoke a callback for adding dynamic args
- **cli**: provide choices for dataset converters
- **cli**: prepare and exec parsers on subcommands
- **cli**: add a base class for cli subcommands
- **cli**: add convert dataset cli command
- **cli**: add a pipeline for training tokenizers
- **cli**: add a script to pretokenize python source files
- **cli**: add a select option to build datasets of different types
- **cli**: add data preprocess cli script to convert text datasets to indexed datasets

# Base Transformers

- **base_transformers**: save hyperparameters to restore args from checkpoints
- **base_transformers**: add perplexity metric basic impl
- **base_transformers**: add protocols with properties decl for type hints
- **base_transformers**: add learning_rate_end param
- **base_transformers**: inject code special tokens while training a tokenizer
- **base_transformers**: make output dirs if not exists
- **base_transformer**: inject code tokenizer special tokens
- **base_transformers**: add tokenizer required args
- **base_transformers**: define APIs for tokenization module and trainer
- **base_transformers**: add a transformer robertafast-like tokenizer
- **base_transformers**: accept user-defined args in code
- **base_transformers**: add a task class that builds modules and tokenizer
- **base_transformers**: add a mixin to init objects with args dicts
- **base_transformers**: add a base modular transformer tokenizer trainer
- **base_transformers**: calculate total training steps for epoch setting
- **base_transformers**: add an early stopping callback
- **base_transformers**: add wandb logger to track training logs
- **base_transformers**: add deterministic mode with random seed
- **base_transformers**: add an arg for num of checkpoints to keep
- **base_transformers**: register a persistent buffer for best metrics monitor
- **base_transformers**: define arg parser arguments in components
- **base_transformers**: add a callback that saves checkpoints every n steps
- **base_transformers**: use dataset iterator instead of batch samplers
- **base_transformers**: add a data module for preparing the datasets
- **base_transformers**: make indexed dataset impl a required arg
- **base_transformers**: add an option to utilize max tokens batch sampler
- **base_transformers**: add a draft for base trainer that loads comps from configs and cmd
- **base_transformers**: add a method to parse tokenizer from config
- **base_transformers**: make a model from yaml config file
- **base_transformers**: add a module for training language models
- **base_transformers**: add base mixin and params classes for transformer modules

# Common

- **common**: list choices from available comps if a registrable passed
- **common**: add a method to import the user module from dir path
- **common**: add classes for handling static dataclass params
- **common**: add a method to get params of a certain type
- **common**: add utils for package importing
- **common**: add a method to list all registered components
- **common**: add a method to get an init callback from name
- **common**: add a method to find attr in parsed params
- **common**: register dataclass type for new arguments set if not yet
- **common**: add a class for parsing dataclass arguments
- **common**: add dataclass argparse support with static typed fields
- **common**: add constructor parameter for registry records
- **common**: add a registry container for subclasses

# Data

- **data**: add block_size static arg
- **data**: split dataset into train/valid/test subsets
- **data**: add an option for train/test split while converting a dataset
- **data**: search data_files if a path pattern is used
- **data**: add code dataset converter impl
- **data**: add dataset converter base class
- **data**: add a method that inferes the dataset type on reading
- **data**: add a dataset iterator with uniform batch sampling
- **data**: add a class to prepare indexed dataset setup from an impl choice
- **data**: add magic_code to indexed datasets for sanity checks on decoding
- **data**: add memmap indexed dataset implementation
- **data**: add a cached version of indexed dataset
- **data**: add supports_prefetch property for indexed datasets
- **data**: declare data_source attribute
- **data**: return iterator with next method in batch sampler
- **data**: add a special batch sampler for distributed data loaders
- **data**: export BatchSampler base class explicitly
- **data**: add MaxTokensBatchSampler that samples batches of the same length
- **data**: add binarizer class for mapping text to token ids
- **data**: add indexed dataset base class and builder helper class
- **data**: init module

# Utils

- **utils**: add iter_stide method to make overlaping groups from iterable obj
- **utils**: add a method to parse string inputs into boolean values
- **utils**: add a decorator for initializing with dict args for fixed params
- **utils**: add a method to find all subclasses of a given one
- **utils**: add a method to convert Path instance to posix path string
- **utils**: add transcoder python code tokenizer
- **utils**: add a python code2ast parser for getting statements with ast from code
- **utils**: add git repo extractor for cloning repos
- **utils**: add git commit parser class
- **utils**: add utils functions for data processing and metrics calc

# Codebert

- **codebert**: build batch sampler with tpu support
- **codebert**: import torch_xla module if possible for tpu support
- **codebert**: use PreTrainedTokenizerFast abstractions for tokenizer
- **codebert**: add newline token to special tokens
- **codebert**: add lang model pretraining script
- **codebert**: add custom codebert tokenizer subclass
- **codebert**: add text dataset class that reads lines and converts to token ids

# Optim

- **optim**: add AdamW optimizer with correct type hints
- **optim**: add a method to configure params with weight decay
- **optim**: add custom get_polynomial_decay_with_warmup lr scheduler

# Samplers

- **samplers**: add a uniform max tokens batch sampler class
- **samplers**: add base class BatchSampler for batch sampling subclasses

# Misc

- setup main app logging
- add a main file for app setup
- **tasks**: initialize module
- **src**: initialize root module
- **changelog**: add custom cz component for changelog generation
