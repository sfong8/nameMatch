import pandas as pd
import jellyfish

importer_list = pd.read_csv(r'C:\Users\S\PycharmProjects\CompanyNames\HMRC\importsNames.csv')

importer_names = importer_list[['NAME']].drop_duplicates()

# sample_df = pd.read_csv(r'C:\Users\S\PycharmProjects\CompanyNames\data\raw\company_names.csv')
#
#
# x= pd.merge(sample_df,importer_names,how='inner',left_on = ['CompanyName'],right_on=['NAME'])
# x=x[['NAME']].sample(100)
# x.to_csv('matched.csv',index=None )

x= pd.read_csv(r'./HMRC/matched.csv')
y=x['NAME'][0]


z =  [[i,jellyfish.jaro_similarity(i,y)] for i in x['NAME'] if y!=i]
z3 =  [[i,jellyfish.match_rating_comparison(i,y)] for i in x['NAME'] if y!=i]
z2= pd.DataFrame(z)