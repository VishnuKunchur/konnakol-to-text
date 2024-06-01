# fine-tune whisper model to predict konnakol text sequences
import torch
from datasets import load_dataset, Audio # type: ignore
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from utils_konnakol_to_text import prepare_dataset, filter_inputs, filter_labels
from utils_konnakol_to_text import DataCollatorSpeechSeq2SeqWithPadding
from utils_konnakol_to_text import compute_metrics
from utils_konnakol_to_text import WHISPER_SAMPLING_RATE, MAX_DURATION_IN_SECONDS, MAX_LABEL_LENGTH, MODEL_NAME

OUTPUT_DIR = f"./{MODEL_NAME.split('/')[-1]}-konnakol-test"

print(f"""
FINE-TUNING PARAMETERS FOR WHISPER-KONNAKOL:
          
* INPUT AND OUTPUT MAX FEATURE DIMENSIONS *

WHISPER_SAMPLING_RATE = {WHISPER_SAMPLING_RATE}
MAX_DURATION_IN_SECONDS = {MAX_DURATION_IN_SECONDS} 
MAX_LABEL_LENGTH = {MAX_LABEL_LENGTH}
MODEL_NAME = {MODEL_NAME}
OUTPUT_DIR = {OUTPUT_DIR}
""")

print('UNPACKING MODEL COMPONENTS..')
# feature_extractor, processor, model
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language='English', task='transcribe')
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language='English', task='transcribe')

print('LOADING DATASETS..')
# load audio files as a HF dataset
dataset = load_dataset("audiofolder", data_dir="./data")
dataset = dataset.cast_column("audio", Audio(sampling_rate=WHISPER_SAMPLING_RATE))

print('PREPARING DATASET..')
# extract input_features and get labels for all audio files in the dataset
dataset = dataset.map(prepare_dataset, num_proc=1)

print('FILTERING DATA..')
# apply filters:
dataset = dataset.filter(filter_inputs, input_columns=['input_length'])
dataset = dataset.filter(filter_labels, input_columns=['labels_length'])

print('LOADING AND CONFIGURING GENERATIVE MODEL..')
# load model
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
# model config
model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.generation_config.max_length = MAX_LABEL_LENGTH

print(dataset)

# DataCollator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

"""
Seq2Seq Training Arguments
"""
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy='steps',
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=MAX_LABEL_LENGTH,
    save_steps=500,
    eval_steps=100,
    logging_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)
print(f'saved processor to {training_args.output_dir}')

print('training whisper-small-konnakol...')
# train model
trainer.train()
print('** whisper-small-konnakol training complete **')

# save model to path
trainer.save_model(OUTPUT_DIR)











"""
# tests
train_sample = prepare_dataset(dataset["train"][0])
print(train_sample["labels"])
print(train_sample["transcription"])
print("**\nTOKENIZER DECODING\n**")
print(tokenizer.decode(train_sample["labels"]))
"""