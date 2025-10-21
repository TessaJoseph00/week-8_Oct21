"""
apputil.py
A simple Markov chain–based text generator.

Implements:
- get_term_dict(): builds a transition dictionary of words.
- generate(): produces random text based on the Markov property.

This file can be imported as a module or run directly to test.
"""

import re
import random
import numpy as np
import requests
from collections import defaultdict


class MarkovText:
    """
    A simple Markov text generator that learns word transitions
    from a given corpus of text and can generate random sentences.
    """

    def __init__(self, corpus: str) -> None:
        """
        Initialize the MarkovText object.

        Args:
            corpus (str): The input text corpus.
        """
        self.corpus = corpus
        self.term_dict = None

    def get_term_dict(self) -> dict:
        """
        Build a transition dictionary mapping each token (word)
        to the list of words that follow it in the corpus.

        Returns:
            dict: A dictionary of {word: [list of next words]}.
        """
        term_dict = defaultdict(list)

        # Tokenize the corpus into words, ignoring punctuation
        tokens = re.findall(r"\b\w+\b", self.corpus)

        # Build the transition mapping
        for i in range(len(tokens) - 1):
            current_word = tokens[i]
            next_word = tokens[i + 1]
            term_dict[current_word].append(next_word)

        self.term_dict = term_dict
        return term_dict

    def generate(self, seed_term: str | None = None, term_count: int = 15) -> str:
        """
        Generate text using the Markov property.

        Args:
            seed_term (str, optional): Word to start the generation with.
                                       If None, a random word is selected.
            term_count (int): Number of words to generate.

        Returns:
            str: Generated text.
        """
        if self.term_dict is None:
            raise ValueError("Run get_term_dict() before generating text.")

        tokens = list(self.term_dict.keys())

        # Choose a starting word
        if seed_term is None:
            current_word = random.choice(tokens)
        else:
            if seed_term not in self.term_dict:
                raise ValueError(f"'{seed_term}' not found in corpus.")
            current_word = seed_term

        generated = [current_word]

        # Generate subsequent words
        for _ in range(term_count - 1):
            next_words = self.term_dict.get(current_word)

            # Handle case where word has no followers
            if not next_words:
                current_word = random.choice(tokens)
                generated.append(current_word)
                continue

            # Randomly choose the next word
            current_word = np.random.choice(next_words)
            generated.append(current_word)

        return " ".join(generated)

if __name__ == "__main__":
    URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/text/inspiration_quotes.txt"
    content = requests.get(URL, timeout=10)
    quotes_raw = content.text

    # Clean up the quotes text
    quotes = quotes_raw.replace("\n", " ")
    quotes = re.split("[“”]", quotes)
    quotes = quotes[1::2]  # skip non-quote lines
    corpus = " ".join(quotes)
    corpus = re.sub(r"\s+", " ", corpus).strip()

    # Create and test the Markov text generator
    text_gen = MarkovText(corpus)
    text_gen.get_term_dict()

    print("\n--- Randomly Generated Quote ---\n")
    print(text_gen.generate(term_count=20))

    print("\n--- Starting with 'Life' ---\n")
    print(text_gen.generate(seed_term="Life", term_count=25))

