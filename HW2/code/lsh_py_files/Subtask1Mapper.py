#!/bin/python3
import sys


def _write(key, value):
    sys.stdout.write(f"{key}\t{value}\n")


def ngrams(words, n=3):
    return [tuple(words[i: i + n]) for i in range(len(words) - n + 1)]


def _map(key, value):
    for title in value:
        words = list(filter(lambda e: len(e), title.split(" ")))  # To remove blank token
        for ngram in ngrams(words):
            _write(key, ",".join(ngram))  # directly print for set?


for line in sys.stdin:
    tokens = line.strip().split("\t")
    # key, value = tokens[0], tokens[1:]
    _map(tokens[0], tokens[1:])
