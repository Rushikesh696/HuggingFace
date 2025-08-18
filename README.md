# Introduction to Hugging Face

This repository is a hands-on guide to exploring Hugging Face’s Transformers, Datasets, and Pipelines. It covers a wide range of NLP and Computer Vision tasks, starting from the basics of pipelines to working with datasets, tokenizers, and models.

**What is Hugging Face?**

Hugging Face provides an ecosystem of tools and libraries that make it easy to use state-of-the-art machine learning models. With the transformers library, you can access pretrained models for NLP, vision, and multimodal tasks, while the datasets library gives access to hundreds of ready-to-use datasets.

**Covered Topics*
Natural Language Processing (NLP) with Pipelines

Text Classification – Sentiment analysis, topic classification.

Named Entity Recognition (NER) – Extracting entities like names, dates, and locations.

Summarization – Generating concise summaries from long texts.

Question Answering – Extracting answers from context passages.

Translation – Translating text between languages.

Text Generation – Autoregressive text generation with GPT-style models.

Fill-Mask – Predicting masked tokens in a sentence.

Feature Extraction – Extracting hidden state embeddings from transformer models.

**Computer Vision with Pipelines**

Image Classification – Identifying objects in an image.

Object Detection – Detecting objects with bounding boxes.

Image Segmentation – Pixel-level classification of images.

**Datasets, Models, and Pipelines**

This repo also explores Hugging Face’s Datasets and Transformers libraries to build workflows beyond just pipelines.

XSum Dataset

The XSum (Extreme Summarization) dataset is designed for abstractive summarization tasks.

It contains BBC news articles with single-sentence summaries.

Used here for experimenting with summarization pipelines.

**AutoTokenizer**

AutoTokenizer automatically loads the correct tokenizer for any given model.

Example:

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")

**AutoModel**

AutoModel and its variants load pretrained models for different tasks.

Example:

from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")

**Hugging Face Pipelines with XSum**

End-to-end summarization with BART on XSum dataset:

from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-xsum")
summarizer("The BBC reported new updates about...", max_length=50, min_length=10)

**Key Learnings**

How to use Hugging Face pipelines for quick experimentation.

How to load and work with datasets like XSum.

How to tokenize text using AutoTokenizer.

How to load pretrained AutoModels for different tasks.

How to run end-to-end workflows combining datasets, models, and pipelines.

This repo serves as a starter guide for anyone who wants to dive into Hugging Face and explore the powerful tools it provides for NLP and Computer Vision.
