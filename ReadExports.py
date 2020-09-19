import pandas as pd
# col_specs = [(1,9),	(10,10),	(11,15),	(16,16),	(17,19),	(20,20),	(21,23),	(24,24),	(25,26),	(27,27),	(28,29),	(30,30),	(31,34),	(35,35),	(36,38),	(39,39),	(40,42),	(43,43),	(44,46),	(47,47),	(48,49),	(50,50),	(51,51),	(52,52),	(53,55),	(56,56),	(57,59),	(60,60),	(61,62),	(63,63),	(64,66),	(67,67),	(68,70),	(71,71),	(72,74),	(75,75),	(76,78),	(79,79),	(80,95),	(96,96),	(97,110),	(111,111),	(112,125),	(126,126),	(127,141)]
#x= pd.read_fwf('SMKE192006',header=None,colspecs=col_specs,dtype=str)

import os
filepath = r'C:\Users\S\PycharmProjects\nameMatch\data\NonEU_Exports'
master_df = pd.DataFrame()


for file in os.listdir(filepath):
    x= pd.read_csv(fr'{filepath}//{file}',skiprows=1,header=None, dtype=str,sep='|')
    mmyyyy= file[8:10]+'/20'+file[6:8]
    cols = ['MAF-COMCODE',		'MAF-SITC',		'MAF-RECORD-TYPE',		'MAF-COD-SEQUENCE',		'MAF-COD-ALPHA',		'MAF-ACCOUNT-MMCCYY',		'MAF-PORT-SEQUENCE',		'MAF-PORT-ALPHA',		'MAF-FLAG-SEQUENCE',		'MAF-FLAG-ALPHA',		'MAF-TRADE-INDICATOR',		'MAF-CONTAINER',		'MAF-MODE-OF-TRANSPORT',		'MAF-INLAND-MOT',		'MAF-GOLO-SEQUENCE',		'MAF-GOLO-ALPHA',		'MAF-SUITE-INDICATOR',		'MAF-PROCEDURE-CODE',		'MAF-VALUE',		'MAF-QUANTITY-1',		'MAF-QUANTITY-2',		'MAF-INDUSTRIAL-PLANT-COMCODE']
    x.columns = cols

    import pycountry

    x1=x[x['MAF-RECORD-TYPE']=='000']
    x1=x1[x1['MAF-ACCOUNT-MMCCYY']==mmyyyy]
    #list_alpha_2 = [{i.alpha_2,i.name} for i in list(pycountry.countries)]

    list_alpha_2={i.alpha_2: i.name for i in list(pycountry.countries)}
    x1['COUNTRY']=x1['MAF-COD-ALPHA'].apply(lambda y:list_alpha_2.get(y))

    x2 = x1[['MAF-COMCODE','MAF-ACCOUNT-MMCCYY','COUNTRY','MAF-VALUE']]
    x2.columns = ['COMCODE','MMYYY','DESTINATION_COUNTRY','VALUE']
    x2['VALUE']=x2['VALUE'].apply(lambda  x: int(str(x[1:])))
    x2_grouped = x2.groupby(['COMCODE','MMYYY','DESTINATION_COUNTRY']).sum().reset_index()
    master_df=pd.concat([master_df,x2_grouped])

master_df.to_csv('nonEU_exports.csv',index=None )

# master_df2 = master_df[master_df['DESTINATION_COUNTRY'].isna()]