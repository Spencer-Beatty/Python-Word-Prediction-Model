# N-gram Language Model for Word Prediction

This repository contains a Python script that defines an N-gram Language Model, particularly for trigrams, to predict words based on previous word sequences in text. This type of predictive modeling is crucial in Natural Language Processing (NLP) for a range of applications, including autocomplete systems, chatbots, and machine translation.

## Model Overview

The model works by maintaining two primary data structures: 

1. A dictionary to store the possible next words for a given bigram.
2. A dictionary to store the frequency counts of these possible words. 

These frequencies are then used to compute probabilities, encapsulating the likelihood of a word following a specific bigram in the training data set.

## Key Methods

### `load_trigrams()`

This method loads trigrams from a data file and populates the dictionaries with trigrams and their counts.

### `top_next_word(word1, word2, n=10)`

This method takes a bigram as input and returns the top `n` most probable words that could follow this bigram based on the training data. 

### `sample_next_word(word1, word2, n=10)`

Similar to `top_next_word`, this function samples from the entire probability distribution of next words, rather than returning only the top `n` possibilities.

### `generate_sentences(prefix, beam=10, sampler=top_next_word, max_len=20)`

This method employs a beam search strategy to generate sentences of a maximum length `max_len`, starting with a given prefix.

## Usage Example

Consider a model trained on a corpus of tweets about the COVID-19 pandemic. 

- With the input bigram "pandemic is", the `top_next_word` method might return words like "over", "not", and "worsening" with their respective probabilities. These are the most common words following "pandemic is" in the dataset.
- The `sample_next_word` function might return words like "affecting" and "slowing", also with their respective probabilities.
- Lastly, with a prefix of "<BOS1> <BOS2> trump", the `generate_sentences` method could generate sentences like "<BOS1> <BOS2> trump claims pandemic is over", given a specific beam size and sampler function. 

This script provides a foundational approach to text prediction tasks and can be adapted or extended for various other NLP applications.
