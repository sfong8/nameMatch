import pandas as pd
import jellyfish
import numpy as np
importer_list = pd.read_csv(r'C:\Users\S\PycharmProjects\CompanyNames\HMRC\importsNames.csv')
# importer_list.columns = ['NAME']
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
y= pd.read_csv(r'./HMRC/matched.csv')
def preprocess(text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens = tokens.split()
    return tokens


lsh = MinHashLSH(
    threshold= 0.8, # Jaccard similarity threshold
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


y= pd.read_csv(r'./HMRC/matched.csv')

query_review= y['NAME'][0]
similar_reviews_ids = query(lsh, query_review)
