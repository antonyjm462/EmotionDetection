# imports
import spacy
import nltk
from emoji import demojize
from nltk.tokenize import TweetTokenizer
import re
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
import unicodedata
import pandas as pd 
from pandas import DataFrame, read_csv
import os
import csv 
import numpy as np
special_char_pattern = re.compile(r'([{.(-)!}])')
from nltk.stem import PorterStemmer
ps = PorterStemmer()

nlp = spacy.load('en_core_web_sm', parse = False, tag=False, entity=False)
tokenizer = TweetTokenizer(strip_handles=True)
stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove('not')
stopwords.remove('nor')
stopwords.remove('no')


def rm_urls(doc):
    URLless_string = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', doc)
    return URLless_string

def strip_html_tags(doc):
    soup = BeautifulSoup(doc, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def expand_contractions(doc, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
                                   if contraction_mapping.get(match) \
                                    else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, doc)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# Remove emoji
def rm_special_chars(doc):
    doc = demojize(doc)
    doc = doc.replace(r"::", ": :")
    doc = doc.replace(r"’", "'")
    doc = doc.replace(r"[^a-z\':_]", " ")
    return doc

# Remove repetitions
def rm_repetitions(doc):
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    doc = pattern.sub(r"\1", doc)
    return doc

# Transform short negation form
def short_negation(doc):
    doc = doc.replace(r"(can't|cannot)", 'can not')
    doc = doc.replace(r"n't", ' not')
    return doc

# Remove stop words and stem
def rm_stop_words(doc):
    tokens = tokenizer.tokenize(doc)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopwords]
    filtered_tokens = [ps.stem(token) for token in filtered_tokens]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# Remove special characters  
def remove_special_characters(text):
	text = re.sub('[^a-zA-Z0-9\s]', '', text)
	return text

def rm_after_url(doc):
    i = doc.find('https:')
    while(i != -1):
        url = doc[i:i+25]
        doc = ''.join(doc.split(url))
        i = doc.find('https:')
    return doc


def normalize_corpus(corpus):
    normalized_corpus = []
    
    for doc in corpus:
        doc = rm_urls(doc)
        doc = strip_html_tags(doc)
        doc = doc.lower()
        doc = rm_special_chars(doc)
        doc = rm_repetitions(doc)
        doc = short_negation(doc)
        doc = expand_contractions(doc, contraction_mapping=CONTRACTION_MAP)
        doc = rm_stop_words(doc)
        doc = rm_after_url(doc)
        doc = remove_special_characters(doc)
        normalized_corpus.append(doc)
        
    return normalized_corpus
