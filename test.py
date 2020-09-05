from fuzzywuzzy import fuzz
from fuzzywuzzy import process

s = ['Gonzalez, E. walked down the street.', 'Gonzalez, R. went to the market.', 'Clemens, Ko. reach the intersection; Clemens, Ka. did not.']

names = []

for i in s:

    name = [] #clear name
    for k in i.split():
        if k[0].isupper(): name.append(k)
        else: break
    names.append(' '.join(name))

    if ';' in i:
        for each in i.split(';')[1:]:
            name = [] #clear name
            for k in each.split():
                if k[0].isupper(): name.append(k)
                else: break
            names.append(' '.join(name))

print(names)

choices = ['Kody Clemens','Kacy Clemens','Gonzalez Ryan', 'Gonzalez Eddy']
import pandas as pd
def bestmatch_name(i):
    s1 = process.extract(i, choices,scorer=fuzz.token_set_ratio)
    s2 = process.extract(i, choices, scorer=fuzz.token_sort_ratio)
    s3 = process.extract(i, choices, scorer=fuzz.token_set_ratio)
    s4=s1+s2+s3
    t1 = pd.DataFrame(s4,columns=['Name','Score'])
    #res1 =max(s,key=lambda x:x[1])
    #print(res1, s, i)
    avg_t1 = t1.groupby(['Name']).mean().reset_index()
    return avg_t1.iloc[avg_t1['Score'].idxmax()]['Name'],avg_t1.iloc[avg_t1['Score'].idxmax()]['Score']