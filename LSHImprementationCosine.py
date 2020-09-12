import pandas as pd
import jellyfish
import numpy as np


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

importer_list = pd.read_csv(r'C:\Users\S\PycharmProjects\CompanyNames\HMRC\importsNames.csv')
# importer_list.columns = ['NAME']
importer_list['NAME'] = importer_list['NAME'].apply(lambda x: process_companyName(x))

df = importer_list[['NAME']].drop_duplicates().reset_index(drop=True)
df['id']=df.index
# sample_df = pd.read_csv(r'C:\Users\S\PycharmProjects\CompanyNames\data\raw\company_names.csv')
#
#
# x= pd.merge(sample_df,importer_names,how='inner',left_on = ['CompanyName'],right_on=['NAME'])
# x=x[['NAME']].sample(100)
# x.to_csv('matched.csv',index=None )

y= pd.read_csv(r'./HMRC/matched.csv')
y['NAME'] = y['NAME'].apply(lambda x: process_companyName(x))

import re
def ngrams_analyzer(string):
    string = re.sub(r'[,-./]', r'', string)
    ngrams = zip(*[string[i:] for i in range(5)])  # N-Gram length is 5
    return [''.join(ngram) for ngram in ngrams]


from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),
    min_df=0,
    stop_words='english')
X_tfidf = tfidf.fit_transform(df['NAME'])


def generate_random_vectors(dim, n_vectors):
    """
    generate random projection vectors
    the dims comes first in the matrix's shape,
    so we can use it for matrix multiplication.
    """
    return np.random.randn(dim, n_vectors)


from collections import defaultdict


def train_lsh(X_tfidf, n_vectors, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dim = X_tfidf.shape[1]
    random_vectors = generate_random_vectors(dim, n_vectors)

    # partition data points into bins,
    # and encode bin index bits into integers
    bin_indices_bits = X_tfidf.dot(random_vectors) >= 0
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
    bin_indices = bin_indices_bits.dot(powers_of_two)

    # update `table` so that `table[i]` is the list of document ids with bin index equal to i
    table = defaultdict(list)
    for idx, bin_index in enumerate(bin_indices):
        table[bin_index].append(idx)

    # note that we're storing the bin_indices here
    # so we can do some ad-hoc checking with it,
    # this isn't actually required
    model = {'table': table,
             'random_vectors': random_vectors,
             'bin_indices': bin_indices,
             'bin_indices_bits': bin_indices_bits}
    return model


# train the model
n_vectors = 16
model = train_lsh(X_tfidf, n_vectors, seed=143)



from itertools import combinations


def search_nearby_bins(query_bin_bits, table, search_radius=3, candidate_set=None):
    """
    For a given query vector and trained LSH model's table
    return all candidate neighbors with the specified search radius.

    Example
    -------
    model = train_lsh(X_tfidf, n_vectors=16, seed=143)
    query = model['bin_index_bits'][0]  # vector for the first document
    candidates = search_nearby_bins(query, model['table'])
    """
    if candidate_set is None:
        candidate_set = set()

    n_vectors = query_bin_bits.shape[0]
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)

    for different_bits in combinations(range(n_vectors), search_radius):
        # flip the bits (n_1, n_2, ..., n_r) of the query bin to produce a new bit vector
        index = list(different_bits)
        alternate_bits = query_bin_bits.copy()
        alternate_bits[index] = np.logical_not(alternate_bits[index])

        # convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)

        # fetch the list of documents belonging to
        # the bin indexed by the new bit vector,
        # then add those documents to candidate_set;
        # make sure that the bin exists in the table
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin])

    return candidate_set


from sklearn.metrics.pairwise import pairwise_distances


def get_nearest_neighbors(X_tfidf, query_vector, model, max_search_radius=3):
    table = model['table']
    random_vectors = model['random_vectors']

    # compute bin index for the query vector, in bit representation.
    bin_index_bits = np.ravel(query_vector.dot(random_vectors) >= 0)

    # search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in range(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, candidate_set)

    # sort candidates by their true distances from the query
    candidate_list = list(candidate_set)
    candidates = X_tfidf[candidate_list]
    distance = pairwise_distances(candidates, query_vector, metric='cosine').flatten()
    distance2 = pairwise_distances(candidates.todense(), query_vector.todense(), metric='jaccard').flatten()

    distance_col = 'distance'
    nearest_neighbors = pd.DataFrame({
        'id': candidate_list, distance_col: distance
    }).sort_values(distance_col).reset_index(drop=True)
    return nearest_neighbors


y_tfidf = tfidf.transform(y['NAME'])
master_df = pd.DataFrame()


for i in range(y.shape[0]):

    query_vector = y_tfidf[i]
    nearest_neighbors = get_nearest_neighbors(X_tfidf, query_vector, model, max_search_radius=5)
    print('dimension: ', nearest_neighbors.shape)
    nearest_neighbors.head()
    nearest_neighbors['ActualName']=y['NAME'][i]
    master_df=pd.concat([master_df,nearest_neighbors.head(1)])
    print(y['NAME'][i],df['NAME'][nearest_neighbors.head(1)['id']])





# we can perform a join with the original table to get the description
# for sanity checking purpose
master= master_df.merge(df, on='id', how='left')