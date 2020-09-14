import pandas as pd
import numpy as np
names_dict = {'LTD':'LIMITED','(':'', ')':'', ',':'', '.':'', '-':' ', '  ':' ',"'":'','&':' AND ','LTD':'','LIMITED':'','-':' '}

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




# importer_list = pd.read_csv(r'C:\Users\S\PycharmProjects\CompanyNames\HMRC\importsNames.csv')
importer_list = pd.read_csv(r'./Data/company_names.csv')

importer_list.columns = ['NAME']
importer_list['NAME'] = importer_list['NAME'].apply(lambda x: process_companyName(x))
importer_list=importer_list.sample(frac=0.33)
df = importer_list[['NAME']].drop_duplicates().reset_index(drop=True)
df['id']=df.index
# sample_df = pd.read_csv(r'C:\Users\S\PycharmProjects\CompanyNames\data\raw\company_names.csv')
#
#
# x= pd.merge(sample_df,importer_names,how='inner',left_on = ['CompanyName'],right_on=['NAME'])
# x=x[['NAME']].sample(100)
# x.to_csv('matched.csv',index=None )
import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest,MinHashLSH
from tqdm import tqdm
# y= pd.read_csv(r'C:/Users/S/PycharmProjects/CompanyNames/HMRC/matched.csv')
y= pd.read_csv(r'./Data/matched.csv')
y['NAME'] = y['NAME'].apply(lambda x: process_companyName(x))

def preprocess(text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens = tokens.split()
    return tokens


lsh = MinHashLSH(
    threshold= 0.5, # Jaccard similarity threshold
    num_perm=128,   # Number of hash functions
)


reviews = dict()
for key, value in enumerate(df.NAME.values):
    reviews[key] = value

for key in tqdm(reviews):
    review_hash = MinHash()
    # Iterate over words and add to the review specific hash
    for word in reviews[key].split(" "):
        review_hash.update(word.encode("utf-8"))

    # Now insert the review specific hash to lsh object
    lsh.insert(key, review_hash)


def query(lsh, sentence):
    """ builds the hash for the sentence and query's the hash for similar senteces """
    sentence_hash = MinHash()
    for word in sentence.split(" "):
        sentence_hash.update(word.encode("utf-8"))

    similar_sentences_ids = lsh.query(sentence_hash)
    return similar_sentences_ids





def compute_jaccard(sentence1, sentence2):
    words_in_sentence1 = set(sentence1.split(" "))
    words_in_sentence2 = set(sentence2.split(" "))

    return len(words_in_sentence1 & words_in_sentence2) / len(words_in_sentence1 | words_in_sentence2)


master_df = pd.DataFrame()

for i in range(y.shape[0]):
    query_review = y['NAME'][i]
    similar_reviews_ids = query(lsh, query_review)
    similar_reviews_ids2 = pd.DataFrame(query(lsh, query_review),columns=['id'])
    similar_reviews_ids2['NAME']=query_review
    similar_reviews_ids2['MATCHED_NAME']=similar_reviews_ids2.id.apply(lambda x : df['NAME'][x])
    similar_reviews_ids2['score']=similar_reviews_ids2.id.apply(lambda x : compute_jaccard(query_review,df['NAME'][x]))
    similar_reviews_ids2=similar_reviews_ids2.sort_values(['score'],ascending=False)
    master_df=pd.concat([master_df,similar_reviews_ids2.head(1)])