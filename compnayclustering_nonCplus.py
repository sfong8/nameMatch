import pandas as pd
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
from sklearn.metrics.pairwise import cosine_similarity
# Instaniate our lookup hash table
group_lookup = {}


names_dict = {'LTD':'LIMITED','(':'', ')':'', ',':'', '.':'', '-':' ', '  ':' ',"'":'','&':' AND '}

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
    new_str =  str.upper(cName)
    new_str = replace_all(names_dict,new_str)
    new_str = newTradingAs(new_str,trading_as_list)
    return ' '.join(new_str.split())

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
    ngrams = zip(*[string[i:] for i in range(2)])  # N-Gram length is 2
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

###function to get the cosine similarty score (above a certain threshold) and return the unique values indexs
def getCosineSim(tfidf,min_threshold=0.8):
    cosine_df = pd.DataFrame()
    for x in range(len(vals)):
        cossim_matrix = cosine_similarity(tfidf_matrix[x], tfidf_matrix).flatten()
        z=[[x,i, cossim_matrix[i]] for i in range(cossim_matrix.__len__()) if cossim_matrix[i]>min_threshold]
        cosine_df=pd.concat([cosine_df,pd.DataFrame(z)])
    cosine_df.columns = ['val_index','val_index2','cosineSim_score']
    return cosine_df


# Grab the column you'd like to group, filter out duplicate values
# and make sure the values are Unicode
vectorizer = TfidfVectorizer(analyzer=ngrams_analyzer)
vals = df['NewName'].unique().astype('U')

# Build the matrix!!!
tfidf_matrix = vectorizer.fit_transform(vals)


###

cosine_df=getCosineSim(tfidf=tfidf_matrix,min_threshold=0.8)
for row, col in zip(cosine_df.val_index, cosine_df.val_index2):
    if row != col:
        add_pair_to_lookup(vals[row], vals[col])
df['Group'] = df['NewName'].map(group_lookup).fillna(df['NewName'])
print(df['Group'].isna().sum())

df.to_csv('test_nonCplus.csv',index=None)