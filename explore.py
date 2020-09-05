import pandas as pd

# data = pd.read_csv(r'C:\Users\S\PycharmProjects\CompanyNames\data\raw\Co_file.csv')
#company_names = data[['CompanyName']]
#company_names.to_csv(r'./data/raw/company_names.csv',index=None)
company_names=pd.read_csv(r'./data/raw/company_names.csv')
#company_names=company_names.sample(frac=0.8)
#
#
# sample=data.sample(10000)
#
# sample=pd.read_csv(r'C:\Users\S\PycharmProjects\CompanyNames\data\raw\sample.csv')
# sample_100 = sample.sample(100)
# sample_100.to_csv(r'C:\Users\S\PycharmProjects\CompanyNames\data\raw\sample100.csv',index=None)


#df=pd.read_csv(r'/Users/simonfong/PycharmProjects/CompanyNamesMatching/data/raw/test.csv')
df=pd.read_csv(r'./data/raw/test.csv')
sample=pd.read_csv(r'./data/raw/sample.csv')
uniq_companyName = pd.DataFrame(df['CompanyName'].unique())
x=df[df.CompanyNumber=='LP008789']

import fuzzywuzzy

names_dict = {'LTD':'LIMITED','(':'', ')':'', ',':'', '.':'', ' - ':' ', '  ':' ',"'":''}

trading_as_list = ['T/AS ','T/A ','TRADING AS','TRADINGAS', 'T / A ',]

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
    return new_str


# q = [j for i in trading_as_list for j in  sample_companyNames if i in j]
# q = [j for j in  sample_companyNames if '&' in j]
#
#
# q_df = pd.DataFrame(q)
#
#
# new_y = newTradingAs(y,trading_as_list)
#
# test_str = 'asdfa (test) ltd'
#
# companyname = str.upper(test_str)
# companyname2 = replace_all(names_dict,companyname)
df=df[['CompanyName','CompanyNumber']]
df['processName']=df['CompanyName'].apply(lambda x : process_companyName(x))


def company_Initials(companyname):
    new_str=''
    for i in companyname.split():
        if i in ['LIMITED','COMPNAY','&', 'AND','CO','LLP','INVESTMENTS','INVESTMENT']:
            break
        new_str+=i[0]
    return new_str 

df['iniitals']=df['processName'].apply(lambda x : company_Initials(x))

x= df[['processName','CompanyNumber']].groupby(['processName','CompanyNumber']).size().reset_index()
x.columns = ['processName','CompanyNumber','Size']
###sort it by the size in descending order
x=x.sort_values(['Size'],ascending=False).reset_index(drop=True)

###want to only keep the highest value of
x2= x.drop_duplicates(subset = ['CompanyNumber']).reset_index(drop=True)
more_than1=  list(x2['processName'])
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def bestmatch_name(i):
    s1 = process.extract(i, more_than1,scorer=fuzz.token_set_ratio)
    s2 = process.extract(i, more_than1, scorer=fuzz.token_sort_ratio)
    s3 = process.extract(i, more_than1, scorer=fuzz.token_set_ratio)
    s4=s1+s2+s3
    t1 = pd.DataFrame(s4,columns=['Name','Score'])
    #res1 =max(s,key=lambda x:x[1])
    #print(res1, s, i)
    avg_t1 = t1.groupby(['Name']).mean().reset_index()
    return avg_t1.iloc[avg_t1['Score'].idxmax()]['Name'],avg_t1.iloc[avg_t1['Score'].idxmax()]['Score']

df['bestMatch_name'] = df['processName'].apply(lambda x : bestmatch_name(x)[0])
df['bestMatch_ratio'] = df['processName'].apply(lambda x : bestmatch_name(x)[1])