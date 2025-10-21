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
    A Markov text generator that learns word transitions
    from a given corpus and generates random sentences.
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
        Build a transition dictionary mapping each word (state)
        to the list of words that follow it in the corpus.

        Returns:
            dict: A dictionary where keys are words and values are
                  lists of words that can follow the key.
        """
        term_dict = defaultdict(list)

        # Tokenize the corpus into words, ignoring punctuation
        tokens = re.findall(r"\b\w+\b", self.corpus)

        # Construct the mapping of each word to its subsequent words
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
            seed_term (str, optional): Word to start the generation.
                                       If None, a random word is selected.
            term_count (int): Number of words to generate.

        Returns:
            str: Generated text sequence.
        """
        if self.term_dict is None:
            raise ValueError("Run get_term_dict() before generating text.")

        tokens = list(self.term_dict.keys())

        # Choose starting word
        if seed_term is None:
            current_word = random.choice(tokens)
        else:
            if seed_term not in self.term_dict:
                raise ValueError(f"'{seed_term}' not found in corpus.")
            current_word = seed_term

        generated = [current_word]

        # Generate subsequent words iteratively
        for _ in range(term_count - 1):
            next_words = self.term_dict.get(current_word)

            # If current word has no followers, pick a random word
            if not next_words:
                current_word = random.choice(tokens)
                generated.append(current_word)
                continue

            # Randomly select the next word 
            current_word = np.random.choice(next_words)
            generated.append(current_word)

        return " ".join(generated)


class MarkovTextK:
    """
    Bonus: Markov text generator using a k-word state window.
    """

    def __init__(self, corpus: str, k: int = 1) -> None:
        """
        Initialize the MarkovTextK object.

        Args:
            corpus (str): Input text corpus.
            k (int): Number of words per state (window size).
        """
        self.corpus = corpus
        self.k = k
        self.term_dict = None

    def get_term_dict(self) -> dict:
        """
        Build a transition dictionary with k-word keys mapping
        to a list of subsequent words.

        Returns:
            dict: Mapping of k-word tuples to lists of possible next words.
        """
        term_dict = defaultdict(list)

        # Tokenize the corpus
        tokens = re.findall(r"\b\w+\b", self.corpus)

        # Build k-word state mapping
        for i in range(len(tokens) - self.k):
            key = tuple(tokens[i:i + self.k])
            next_word = tokens[i + self.k]
            term_dict[key].append(next_word)

        self.term_dict = term_dict
        return term_dict

    def generate(self, seed_term: str | None = None, term_count: int = 15) -> str:
        """
        Generate text using k-word Markov states.

        Args:
            seed_term (str, optional): Word to start the generation.
            term_count (int): Number of words to generate.

        Returns:
            str: Generated text sequence.
        """
        if self.term_dict is None:
            raise ValueError("Run get_term_dict() before generating text.")

        tokens = list(self.term_dict.keys())

        # starting state
        if seed_term is None:
            current_state = random.choice(tokens)
        else:
            # Convert seed_term into tuple for k-word state
            seed_tokens = re.findall(r"\b\w+\b", seed_term)
            if len(seed_tokens) != self.k:
                raise ValueError(f"Seed term must have exactly {self.k} words.")
            current_state = tuple(seed_tokens)
            if current_state not in self.term_dict:
                raise ValueError(f"'{seed_term}' not found in corpus as a k-word state.")
        
        generated = list(current_state)

        # Generate subsequent words iteratively
        for _ in range(term_count - self.k):
            next_words = self.term_dict.get(current_state)
            if not next_words:
                current_state = random.choice(tokens)
                generated.extend(list(current_state))
                continue

            next_word = np.random.choice(next_words)
            generated.append(next_word)

            current_state = tuple(generated[-self.k:])

        return " ".join(generated)


if __name__ == "__main__":
    URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/text/inspiration_quotes.txt"
    content = requests.get(URL, timeout=10)
    quotes_raw = content.text

    # Extract quotes from raw text
    quotes = quotes_raw.replace("\n", " ")
    quotes = re.split("[“”]", quotes)
    quotes = quotes[1::2]  # select every other element (actual quotes)

    # Combine all quotes into a single corpus string
    corpus = " ".join(quotes)
    corpus = re.sub(r"\s+", " ", corpus).strip()  # clean up whitespace

    # Test 1-word Markov generator
    text_gen_1 = MarkovText(corpus)
    text_gen_1.get_term_dict()
    print("\n--- Randomly Generated (1-word model) ---\n")
    print(text_gen_1.generate(term_count=20))

    # Test 2-word Markov generator
    text_gen_2 = MarkovTextK(corpus, k=2)
    text_gen_2.get_term_dict()
    print("\n--- Randomly Generated (2-word model) ---\n")
    print(text_gen_2.generate(term_count=30))

