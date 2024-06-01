# test fine-tuned whisper-small-konnakol
from time import time
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

whisper_model_name = './whisper-small-konnakol'
lesson_audio_filepath = './data/test/ch1_l11.wav'
RECORDED_SAMPLING_RATE = 44_100
WHISPER_SAMPLING_RATE = 16_000

# check if cuda device is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained(whisper_model_name)
# load model
model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

# enforce english transcription, i.e. prevent output language auto-detection
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language='english', task='transcribe')
model = model.to(device)
print(f'{whisper_model_name} load complete on {device}')

# load konnakol audio to be transcribed
audio_array, _ = librosa.load(lesson_audio_filepath, sr=RECORDED_SAMPLING_RATE)
audio_array = librosa.resample(audio_array, orig_sr=RECORDED_SAMPLING_RATE, target_sr=WHISPER_SAMPLING_RATE)

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
print(f'{whisper_model_name} inference in progress..')
predicted_ids = model.generate(input_features, language='en')
# decode tokens to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
end = time()
print(f'inference time: {round(end-sta, 3)}s')










"""
from transcription.hf_whisper_konnakol_tests import predict_konnakol_text
from transcription.hf_whisper_konnakol_tests import konnakol_vocab_test_2
#import random

test_lesson_audio_filepath = './data/chapter-lessons-audio-files/chapter-1/lesson-2.wav'
whisper_model_name='openai/whisper-medium'
RECORDED_SAMPLING_RATE = 44_100
WHISPER_SAMPLING_RATE = 16_000


def test_model(test_lesson_audio_filepath, 
                 whisper_model_name, 
                 RECORDED_SAMPLING_RATE, 
                 WHISPER_SAMPLING_RATE,
                 konnakol_vocab,
                 mode='vanilla'):    
    print(f'MODEL: {whisper_model_name}')
    if mode == 'vanilla':
        print('* Vanilla Konnakol-to-Text Transcription *')
        # load lesson audio data
        predict_konnakol_text(test_lesson_audio_filepath, whisper_model_name, 
                              RECORDED_SAMPLING_RATE, WHISPER_SAMPLING_RATE, suppress_tokens=False)
        pass
    else:
        # predict with token suppression:
        print('* Token-Suppression Konnakol-to-Text Transcription *')
        predict_konnakol_text(test_lesson_audio_filepath, whisper_model_name, 
                              RECORDED_SAMPLING_RATE, WHISPER_SAMPLING_RATE,
                              suppress_tokens=True, konnakol_vocab=konnakol_vocab)

test_model(test_lesson_audio_filepath,
                 whisper_model_name, 
                 RECORDED_SAMPLING_RATE, 
                 WHISPER_SAMPLING_RATE,
                 konnakol_vocab_test_2,
                 mode='vanilla')

# vanilla_test(test_lesson_audio_filepath, whisper_model_name, RECORDED_SAMPLING_RATE, WHISPER_SAMPLING_RATE)

print('success')
"""