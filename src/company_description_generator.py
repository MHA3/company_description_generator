import pandas as pd
import re
import logging
import os
import joblib
import utils
from tqdm import tqdm
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer, BartForConditionalGeneration, BartTokenizer

logging.basicConfig(level=logging.INFO)

class CompanyDescriptionProcessor:
    MAX_SUMMARY_LENGTH = 600
    MIN_SUMMARY_LENGTH = 100
    ESTIMATED_CHARS_PER_TOKEN = 4
    _models = {}

    def __init__(self, batch_size=10, cache_dir='translation_cache_dir'):
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self._load_summarization_models()

    @classmethod
    def _load_translation_models(cls, source_lang):
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-en'
        if model_name not in cls._models:
            cls._models[model_name] = {
                'model': MarianMTModel.from_pretrained(model_name),
                'tokenizer': MarianTokenizer.from_pretrained(model_name)
            }

    @classmethod
    def _load_summarization_models(cls):
        if 'bart_model' not in cls._models:
            cls._models['bart_model'] = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to('cuda')
            cls._models['bart_tokenizer'] = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
            cls.max_tokens = 1024

    def clean_description(self, text):
        cleaned = re.sub('<[^<]+?>', '', text)
        cleaned = re.sub(r'http\S+', '', cleaned)
        cleaned = re.sub(' +', ' ', cleaned)
        return cleaned.strip()

    def get_cache(self, source_lang):
        cache_file = os.path.join(self.cache_dir, f'{source_lang}_cache.pkl')
        if os.path.exists(cache_file):
            return joblib.load(cache_file)
        return {}

    def set_cache(self, cache, source_lang):
        cache_file = os.path.join(self.cache_dir, f'{source_lang}_cache.pkl')
        joblib.dump(cache, cache_file)


    def translate_to_english(self, text, source_lang):
        # If the source language is already English, skip translation
        if source_lang == 'en':
            return text

        cache = self.get_cache(source_lang)
        if text in cache:
            return cache[text]

        try:
            self._load_translation_models(source_lang)
            model = self._models[f'Helsinki-NLP/opus-mt-{source_lang}-en'][
                'model']
            tokenizer = self._models[f'Helsinki-NLP/opus-mt-{source_lang}-en'][
                'tokenizer']

            tokenized_text = tokenizer.encode(text, return_tensors="pt",
                                              truncation=True)
            translation = model.generate(tokenized_text)
            translated_text = tokenizer.decode(translation[0],
                                               skip_special_tokens=True)

            cache[text] = translated_text
            self.set_cache(cache, source_lang)
        except Exception as e:
            logging.warning(
                f"No translation model available for {source_lang}-en. Keeping original text.")
            translated_text = text

        return translated_text


    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return 'en'  # Default to English if detection fails


    def compress(self, text, compressed_length_in_tokens):
        """Perform extractive summarization to reduce the length of the text."""
        compressed_text = text
        max_length = compressed_length_in_tokens * self.ESTIMATED_CHARS_PER_TOKEN
        while len(compressed_text) > max_length:
            text_chunks = utils.chunk_text(compressed_text, max_length)
            prompt = ("Summarize")
            summaries = [self.guided_summarize(s, prompt,
                       compressed_length_in_tokens//2, compressed_length_in_tokens//4)
                       for s in text_chunks]
            compressed_text = ' '.join(summaries)
        return compressed_text

    def guided_summarize(self, text, prefix, max_length, min_length):
        """Generate summary with a specific focus using a prefix."""
        try:
            model = self._models['bart_model']
            tokenizer = self._models['bart_tokenizer']
            input_text = f"{prefix}: {text}"
            input_tokenized = tokenizer.encode(input_text, return_tensors="pt", max_length=self.max_tokens, truncation=True).to('cuda')
            summary_ids = model.generate(input_tokenized, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Error summarizing text with prefix '{prefix}': {e}")
            return ""  # Return empty string if summarization fails

    def process_csv(self, file_path):
        df = pd.read_csv(file_path)
        df['cleaned_description'] = df['text'].apply(self.clean_description)

        if 'language' not in df.columns:
            tqdm.pandas(desc="Detecting Language")
            df['language'] = df['cleaned_description'].progress_apply(self.detect_language)

        tqdm.pandas(desc="Translating")
        df['translated_description'] = df.progress_apply(lambda row: self.translate_to_english(row['cleaned_description'], row['language']), axis=1)

        prompt = "Infer the problem being addressed, the proposed solution and the target audience from this description"
        summaries = []
        compressed_texts = []
        for i in tqdm(range(0, len(df), self.batch_size), desc="Summarizing in batches"):
            compressed_batch = [self.compress(text, self.max_tokens)
                             for text in df['translated_description'].iloc[i:i+self.batch_size]]
            summaries_batch = [self.guided_summarize(text, prompt, self.MAX_SUMMARY_LENGTH, self.MIN_SUMMARY_LENGTH)
                             for text in compressed_batch]
            summaries.extend(summaries_batch)
            compressed_texts.extend(compressed_batch)

        df['summary'] = summaries
        df['compressed_descriptions'] = compressed_texts

        return df[['compressed_descriptions']], df[['summary']]

if __name__ == "__main__":
    file_path = './data/input_sample.csv'
    processor = CompanyDescriptionProcessor()
    compressed_df, summaries_df = processor.process_csv(file_path)
    summaries_df.to_csv("./data/output_sample.csv")
    compressed_df.to_csv("./data/compressed_descriptions.csv")