# imports
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
import unicodedata
import pandas as pd 
from pandas import DataFrame, read_csv
import os
import csv 
import numpy as np 

nlp = spacy.load('en_core_web_sm', parse = False, tag=False, entity=False)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
special_char_pattern = re.compile(r'([{.(-)!}])')

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
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
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_special_characters(text):
	text = re.sub('[^a-zA-Z0-9\s]', '', text)
	text = re.sub(' +', ' ', text)
	return text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def clean(text):
	text = re.sub(r'[\r|\n|\r\n]+', ' ',text) 
	text = special_char_pattern.sub(" \\1 ", text)
	return text

def normalize(data):
	data = strip_html_tags(data)
	data = remove_accented_chars(data)
	data = expand_contractions(data)
	data = data.lower()
	data = clean(data)
	data = lemmatize_text(data)
	data = remove_special_characters(data) 
	data = remove_stopwords(data, is_lower_case=True)
	return data

def imdb_data_preprocess(inpath, outpath, name, mix):
	stopwords = open("stopwords.en.txt", 'r' , encoding="ISO-8859-1").read()
	stopwords = stopwords.split("\n")

	indices = []
	text = []
	rating = []

	i =  0 

	for filename in os.listdir(inpath+"pos"):	
		data = open(inpath+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()
		data,pol = normalize(data)
		indices.append(i)
		text.append(data)
		rating.append(pol)
		i = i + 1

	for filename in os.listdir(inpath+"neg"):
		pol = filename.split("_")
		pol = pol[1].replace(".txt","")
		data = open(inpath+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
		data,pol = normalize(data)
		indices.append(i)
		text.append(data)
		rating.append(pol)
		i = i + 1

	Dataset = list(zip(indices,text,rating))
	
	if mix:
		np.random.shuffle(Dataset)

	df = pd.DataFrame(data = Dataset, columns=['row_number', 'text', 'polarity'])
	df.to_csv(outpath+name, index=False, header=True)

def change_polarity_data(name, train=True):
	import pandas as pd 
	data = pd.read_csv(outpath+name,header=0, encoding = 'ISO-8859-1')
	Y = data['polarity']
	Y = [1 if pol>5 else 0 for pol in Y]
	Dataset = list(zip(data['row_number'],data['text'],Y ))
	df = pd.DataFrame(data = Dataset, columns=['row_number', 'text', 'polarity'])
	df.to_csv("./binary_data/"+name, index=False, header=True)

def main():
    print ("Preprocessing the training_data")
    imdb_data_preprocess(train_path,outpath,"imdb_train.csv", True)
    print ("Preprocessing the testing_data")
    imdb_data_preprocess(test_path,outpath,"imdb_test.csv", True)

if __name__ == "__main__":
    main()
	
	