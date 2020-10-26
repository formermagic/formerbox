"""
This type stub file was generated by pyright.
"""

from transformers import BertModel

"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint."""
def convert_pytorch_checkpoint_to_tf(model: BertModel, ckpt_dir: str, model_name: str):
    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return:

    Currently supported HF models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering
    """
    ...

def main(raw_args=...):
    ...

if __name__ == "__main__":
    ...
