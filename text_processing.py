import re
import nltk
import spacy
import en_core_web_sm
from nltk.tag import pos_tag
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
stemmer = SnowballStemmer("english")
stopwords = stop_words = nltk.corpus.stopwords.words('english')
nlp = en_core_web_sm.load() # loading the language model
analyzer = CountVectorizer().build_analyzer()

def pre_processing_steps(df):
    
    """
    Tokenizes, removes stop words, stems (requires df "texts" labeled)
    Args: dataframe
    Returns: dataframe with processed text
    """
    filtered_tokens=[]
    for doc in df.texts:
        # non_props = strip_proppers_POS(doc)
        token = tokenizer(doc)
        filtered_tokens_temp=[]
        for tok in token:
            if tok not in stop_words:
                clean = clean_str(tok)
                if re.search('[a-zA-Z]', clean):
                    filtered_tokens_temp.append(stemmer.stem(clean))
        filtered_tokens.append(filtered_tokens_temp)
    df['processed_text'] = filtered_tokens    
    return df


def process_text(text):
    '''
    Args: takes text array  
    1. Removes punctuation
    2. Removes stopwords
    Returns: list of clean text words
    '''
    #1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    return clean_words


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

#strip any proper names from a text...unfortunately right now this is yanking the first word from a sentence too.

def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns

# python std preprocessing

def lower(texts):
    return [x.lower() for x in texts]

def depunctuate(texts):
    return [''.join(c for c in x if c not in string.punctuation) for x in texts]

def remove_numbers(texts):
    return [''.join(c for c in x if c not in '0123456789') for x in texts]

def trim_whitespace(texts):
    return [' '.join(x.split()) for x in texts]

# nltk preprocessing

def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# sklearn preprocessing

def stem(doc):
    return (stemmer.stem(w) for w in analyzer(doc))



# spacy preprocessing

def clean_up(text):  # clean up your text and generate list of words for each document. 
    removal=['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE']
    text_out = []
    doc= nlp(text)
    for token in doc:
        if token.is_stop == False and token.is_alpha and len(token)>2 and token.pos_ not in removal:
            lemma = token.lemma_
            text_out.append(lemma)
    return text_out

def preprocess_spacy(data):
    '''
    Args
        data: pandas dataframe with 'texts' column labeled
    Returns: a list of processed texts
    '''
    datalist = data.texts.apply(lambda x:" ".join(clean_up(x)))
    return datalist

def get_doc_term_matrix(data, train_x):
    
    # Create a vocabulary for the lda model and 
    # convert our corpus into document-term matrix for Lda
    datalist = preprocess_spacy(data)
    dictionary = corpora.Dictionary(dataList) 
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in train_x]
    return doc_term_matrix
    