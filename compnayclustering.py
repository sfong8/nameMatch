import pandas as pd
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

# Instaniate our lookup hash table
group_lookup = {}
names_dict = {'LTD':'LIMITED','(':'', ')':'', ',':'', '.':'', ' - ':' ', '  ':' ',"'":'','&':' AND '}

trading_as_list = ['T/AS ','T/A ','TRADING AS','TRADINGAS', 'T / A',]

def newTradingAs(companyname,word_list):
    new_companyName = companyname
    for x in word_list:
        if x in companyname:
            new_companyName = str.strip(companyname.partition(x)[2])
            break
    return new_companyName

def replace_all(dict, str):
    for key in dict:
        str = str.replace(key, dict[key])
    return str

def process_companyName(cName):
    new_str =  str.lower(cName)
    new_str = replace_all(names_dict,new_str)
    new_str = newTradingAs(new_str,trading_as_list)
    return ' '.join(new_str.split())
def stemName(companyName):
    return ' '.join([ps.stem(i) for i in companyName.split()])
# Construct your vectorizer for building the TF-IDF matrix


# Import your data to a Pandas.DataFrame
df = pd.read_csv(r'test.csv')
df=df[['CompanyName']]
df['NewName']=df['CompanyName'].apply(lambda x: process_companyName(x))
#df['stemmedName']=df['NewName'].apply(lambda x: stemName(x))

# Grab the column you'd like to group, filter out duplicate values
# and make sure the values are Unicode
vals = df['NewName'].unique().astype('U')

# Write a function for cleaning strings and returning an array of ngrams
def ngrams_analyzer(string):
    string = re.sub(r'[,-./]', r'', string)
    ngrams = zip(*[string[i:] for i in range(5)])  # N-Gram length is 5
    return [''.join(ngram) for ngram in ngrams]


def find_group(row, col):
    # If either the row or the col string have already been given
    # a group, return that group. Otherwise return none
    if row in group_lookup:
        return group_lookup[row]
    elif col in group_lookup:
        return group_lookup[col]
    else:
        return None


def add_vals_to_lookup(group, row, col):
    # Once we know the group name, set it as the value
    # for both strings in the group_lookup
    group_lookup[row] = group
    group_lookup[col] = group


def add_pair_to_lookup(row, col):
    # in this function we'll add both the row and the col to the lookup
    group = find_group(row, col)  # first, see if one has already been added
    if group is not None:
        # if we already know the group, make sure both row and col are in lookup
        add_vals_to_lookup(group, row, col)
    else:
        # if we get here, we need to add a new group.
        # The name is arbitrary, so just make it the row
        add_vals_to_lookup(row, row, col)


# Grab the column you'd like to group, filter out duplicate values
# and make sure the values are Unicode
vectorizer = TfidfVectorizer(analyzer=ngrams_analyzer)
vals = df['NewName'].unique().astype('U')

# Build the matrix!!!
tfidf_matrix = vectorizer.fit_transform(vals)

cosine_matrix = awesome_cossim_topn(tfidf_matrix, tfidf_matrix.transpose(), vals.size, 0.8)

# Build a coordinate matrix
coo_matrix = cosine_matrix.tocoo()

# for each row and column in coo_matrix
# if they're not the same string add them to the group lookup
for row, col in zip(coo_matrix.row, coo_matrix.col):
    if row != col:
        add_pair_to_lookup(vals[row], vals[col])

df['Group'] = df['NewName'].map(group_lookup).fillna(df['NewName'])

#df.to_csv('./dol-data-grouped.csv')