# functions and utilities to run Whisper fine-tuning and inference

from time import time
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import WhisperForConditionalGeneration
import evaluate

"""
PARAMS
"""
WHISPER_SAMPLING_RATE = 16_000
MAX_DURATION_IN_SECONDS = 30.0
MAX_INPUT_LENGTH = MAX_DURATION_IN_SECONDS * WHISPER_SAMPLING_RATE
# maximum label length is 448 for whisper (output token context length?)
MAX_LABEL_LENGTH = 448
MODEL_NAME = 'openai/whisper-medium'


# feature_extractor, processor, model
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language='English', task='transcribe')
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language='English', task='transcribe')


def prepare_dataset(batch):
    """
    extract input and output features from a dataset instance
    """
    global feature_extractor
    global tokenizer

    # load and resample audio to whisper sampling rate (16khz)
    audio = batch['audio']

    # compute input length
    batch['input_length'] = len(batch['audio'])

    # compute log-Mel input features from input audio array
    batch['input_features'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_features[0]

    # encode target text to label ids
    batch['labels'] = tokenizer(batch['transcription']).input_ids

    # compute labels length **with** special tokens! -> total label length
    batch['labels_length'] = len(batch['labels'])
    return batch

def filter_inputs(input_length):
    """Filter inputs with zero input length or longer than 30s"""
    global MAX_INPUT_LENGTH
    return 0 < input_length < MAX_INPUT_LENGTH

def filter_labels(labels_length):
    """Filter label sequences longer than MAX_LABEL_LENGTH"""
    global MAX_LABEL_LENGTH
    return labels_length < MAX_LABEL_LENGTH

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# model evaluation: WER
def compute_metrics(pred):
    """
    compute Word Error Rate (WER)
    """
    metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
    

# konnakol transcription
def transcribe_konnakol_audio(whisper_model: str, audio_array: np.array, WHISPER_SAMPLING_RATE=16_000):
  """
  convert a konnakol sequence audio array derived from a .wav file (sr = whisper sampling rate)

  whisper_model: str, fine-tuned whisper-konnakol model path
  audio_array: np.array, audio as array
  """
  # check if cuda device is available
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  processor = WhisperProcessor.from_pretrained(whisper_model)
  # load model
  model = WhisperForConditionalGeneration.from_pretrained(whisper_model)

  # enforce english transcription, i.e. prevent output language auto-detection
  model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language='english', task='transcribe')
  model = model.to(device)
  print(f'{whisper_model} load complete on {device}')

  # Whisper expects 30s of audio for shortform transcription
  if len(audio_array) < 30 * WHISPER_SAMPLING_RATE:
      # shortform transcription
      input_features = processor(audio_array, sampling_rate=WHISPER_SAMPLING_RATE, return_tensors='pt').input_features
  else:
      # longform transcription
      input_features = processor(audio_array, sampling_rate=WHISPER_SAMPLING_RATE, return_tensors='pt',
                              truncation=False, padding='longest', return_attention_mask=True).input_features

  input_features = input_features.to(device)
  # GENERATION (MODEL INFERENCE)
  sta = time()
  # generate token ids
  print(f'{whisper_model} inference in progress..')
  predicted_ids = model.generate(input_features, language='en')
  # decode tokens to text
  transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
  # print(transcription)
  end = time()
  print(f'inference time: {round(end-sta, 3)}s')
  transcription = transcription[0]
  return transcription

