import pandas as pd
# col_specs = [(1,9),	(10,10),	(11,15),	(16,16),	(17,19),	(20,20),	(21,23),	(24,24),	(25,26),	(27,27),	(28,29),	(30,30),	(31,34),	(35,35),	(36,38),	(39,39),	(40,42),	(43,43),	(44,46),	(47,47),	(48,49),	(50,50),	(51,51),	(52,52),	(53,55),	(56,56),	(57,59),	(60,60),	(61,62),	(63,63),	(64,66),	(67,67),	(68,70),	(71,71),	(72,74),	(75,75),	(76,78),	(79,79),	(80,95),	(96,96),	(97,110),	(111,111),	(112,125),	(126,126),	(127,141)]
#x= pd.read_fwf('SMKE192006',header=None,colspecs=col_specs,dtype=str)

import os
filepath = r'C:\Users\S\PycharmProjects\nameMatch\data\EUImports'
master_df = pd.DataFrame()
for file in os.listdir(filepath):
    x= pd.read_csv(fr'{filepath}//{file}',skiprows=1,header=None, dtype=str,sep='|')
    x=x[:-1]
    mmyyyy2= '0'+'20'+file[6:8]+file[8:10]
    mmyyyy = file[8:10] + '/20' + file[6:8]
    x1=x[[0,1,3,6,9,14]]
    x1.columns=  ['MAF-COMCODE','MAF-RECORD-TYPE','MAF-COD-ALPHA','MAF-COO-ALPHA','MAF-ACCOUNT-MMCCYY','MAF-VALUE']
                  #cols = ['MAF-COMCODE',		'MAF-SITC',		'MAF-RECORD-TYPE',		'MAF-COD-SEQUENCE',		'MAF-COD-ALPHA',		'MAF-ACCOUNT-MMCCYY',		'MAF-PORT-SEQUENCE',		'MAF-PORT-ALPHA',		'MAF-FLAG-SEQUENCE',		'MAF-FLAG-ALPHA',		'MAF-TRADE-INDICATOR',		'MAF-CONTAINER',		'MAF-MODE-OF-TRANSPORT',		'MAF-INLAND-MOT',		'MAF-GOLO-SEQUENCE',		'MAF-GOLO-ALPHA',		'MAF-SUITE-INDICATOR',		'MAF-PROCEDURE-CODE',		'MAF-VALUE',		'MAF-QUANTITY-1',		'MAF-QUANTITY-2',		'MAF-INDUSTRIAL-PLANT-COMCODE']

    import pycountry

    x1=x1[x1['MAF-RECORD-TYPE']=='0']
    x1=x1[x1['MAF-ACCOUNT-MMCCYY']==mmyyyy2]
    x1['MMYYY']=mmyyyy
    #list_alpha_2 = [{i.alpha_2,i.name} for i in list(pycountry.countries)]

    list_alpha_2={i.alpha_2: i.name for i in list(pycountry.countries)}
    x1['COUNTRY_OF_DISPATCH']=x1['MAF-COD-ALPHA'].apply(lambda y:list_alpha_2.get(y))
    x1['COUNTRY_OF_ORIGIN'] = x1['MAF-COO-ALPHA'].apply(lambda y: list_alpha_2.get(y))

    x2 = x1[['MAF-COMCODE','MMYYY','COUNTRY_OF_DISPATCH','COUNTRY_OF_ORIGIN','MAF-VALUE']]
    x2.columns = ['COMCODE','MMYYY','COUNTRY_OF_DISPATCH','COUNTRY_OF_ORIGIN','VALUE']
    x2['VALUE']=x2['VALUE'].apply(lambda  x: int(str(x[1:])))
    x2['COUNTRY_OF_ORIGIN'] = x2['COUNTRY_OF_ORIGIN'].apply(lambda x: str(x))
    x2_grouped = x2.groupby(['COMCODE','MMYYY','COUNTRY_OF_DISPATCH','COUNTRY_OF_ORIGIN']).sum().reset_index()
    x2_grouped['EU_FLAG']='EU'
    master_df=pd.concat([master_df,x2_grouped])

master_df.to_csv('EU_imports.csv',index=None )

# master_df2 = master_df[master_df['DESTINATION_COUNTRY'].isna()]