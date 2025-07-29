from typing import List
import csv
import datasets

def load_german_sentences(filepath: str) -> List[str]:
    """
    Loads all German sentences from a TSV file into a list of strings.
    Assumes the sentence is in the third column.
    Handles quotes within sentences.
    """
    sentences: List[str] = []
    with open(filepath, mode="r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for row in tsv_reader:
            if len(row) >= 3:
                sentences.append(row[2])
    return sentences


def replace_vowels(sentence: str) -> str:
    vowels = ["a", "e", "i", "o", "u", "ö", "ä", "ü"]
    pairs = ["ei", "ie", "au", "eu", "äu"]
    spoon = ""
    i = 0
    while i < len(sentence):
        if sentence[i].lower() not in vowels:
            spoon += sentence[i]
        else:
            if i + 1 < len(sentence) and sentence[i:i+2].lower() in pairs:
                spoon += sentence[i:i+2] + "lew" + sentence[i:i+2].lower()
                i += 1
            else:
                spoon += sentence[i] + "lew" + sentence[i].lower()
        i += 1

    return spoon

sentences = load_german_sentences("data/ger_sentences.tsv")

dataset = datasets.Dataset.from_dict({"german": sentences, "spoon": [replace_vowels(s) for s in sentences]})

dataset.to_csv("data/german_spoon.csv")