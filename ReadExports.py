import pandas as pd
col_specs = [(1,9),	(10,10),	(11,15),	(16,16),	(17,19),	(20,20),	(21,23),	(24,24),	(25,26),	(27,27),	(28,29),	(30,30),	(31,34),	(35,35),	(36,38),	(39,39),	(40,42),	(43,43),	(44,46),	(47,47),	(48,49),	(50,50),	(51,51),	(52,52),	(53,55),	(56,56),	(57,59),	(60,60),	(61,62),	(63,63),	(64,66),	(67,67),	(68,70),	(71,71),	(72,74),	(75,75),	(76,78),	(79,79),	(80,95),	(96,96),	(97,110),	(111,111),	(112,125),	(126,126),	(127,141)]
#x= pd.read_fwf('SMKE192006',header=None,colspecs=col_specs,dtype=str)
x= pd.read_csv('SMKE192006',skiprows=1,header=None, dtype=str,sep='|')

cols = ['MAF-COMCODE',		'MAF-SITC',		'MAF-RECORD-TYPE',		'MAF-COD-SEQUENCE',		'MAF-COD-ALPHA',		'MAF-ACCOUNT-MMCCYY',		'MAF-PORT-SEQUENCE',		'MAF-PORT-ALPHA',		'MAF-FLAG-SEQUENCE',		'MAF-FLAG-ALPHA',		'MAF-TRADE-INDICATOR',		'MAF-CONTAINER',		'MAF-MODE-OF-TRANSPORT',		'MAF-INLAND-MOT',		'MAF-GOLO-SEQUENCE',		'MAF-GOLO-ALPHA',		'MAF-SUITE-INDICATOR',		'MAF-PROCEDURE-CODE',		'MAF-VALUE',		'MAF-QUANTITY-1',		'MAF-QUANTITY-2',		'MAF-INDUSTRIAL-PLANT-COMCODE']
x.columns = cols
x=x.dropna(axis=1, how='all')

###suppressions
x2=x[x['MAF-RECORD-TYPE']=='002']
x3=x[x['MAF-RECORD-TYPE']=='003']

import pycountry

x1=x[x['MAF-RECORD-TYPE']=='000']

list_alpha_2 = [{i.alpha_2,i.name} for i in list(pycountry.countries)]

list_alpha_2={i.alpha_2: i.name for i in list(pycountry.countries)}
x1['COUNTRY']=x1['MAF-COD-ALPHA'].apply(lambda y:list_alpha_2.get(y))

q= pd.DataFrame(x1['COUNTRY'].value_counts().reset_index())

y1=x[x['MAF-QUANTITY-2']=='+0000000000000']
y2=x[x['MAF-QUANTITY-2']!='+0000000000000']


##load in control files
control_file = pd.read_csv(r'SMKA122007',skiprows=1,header=None, dtype=str,sep='|', encoding= 'unicode_escape')

control_files_columns=['MK-COMCODE',	'MK-INTRA-EXTRA-IND',	'MK-INTRA-MMYY-ON',	'MK-INTRA-MMYY-OFF',	'MK-EXTRA-MMYY-ON',	'MK-EXTRA-MMYY-OFF','MK-NON-TRADE-ID',	'MK-SITC-NO',	'MK-SITC-IND',	'MK-SITC-CONV-A',	'MK-SITC-CONV-B',	'MK-CN-Q2',	'MK-SUPP-ARRIVALS',	'MK-SUPP-DESPATCHES',	'MK-SUPP-IMPORTS',	'MK-SUPP-EXPORTS','MK-SUB-GROUP-ARR'	,'MK-ITEM-ARR',	'MK-SUB-GROUP-DESP',	'MK-ITEM-DESP',	'MK-SUB-GROUP-IMP',	'MK-ITEM-IMP',	'MK-SUB-GROUP-EXP',	'MK-ITEM-EXP',		'MK-QTY1-ALPHA',	'MK-QTY2-ALPHA',	'MK-COMMODITY-ALPHA-2']
control_file.columns=control_files_columns
control_file_qual =control_file[['MK-COMCODE','MK-QTY1-ALPHA',	'MK-QTY2-ALPHA','MK-COMMODITY-ALPHA-2']]

x1_2 = x1.merge(control_file_qual,how='left',left_on='MAF-COMCODE',right_on='MK-COMCODE')

qual1 = x1_2['MK-QTY1-ALPHA'].value_counts().reset_index()
qual2 = x1_2['MK-QTY2-ALPHA'].value_counts().reset_index()

x1_2['qual1']=x1_2['MAF-QUANTITY-1'].apply(lambda x: int(x[1:]))