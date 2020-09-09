import pandas as pd
col_specs = [(1,9),	(10,10),	(11,15),	(16,16),	(17,19),	(20,20),	(21,23),	(24,24),	(25,26),	(27,27),	(28,29),	(30,30),	(31,34),	(35,35),	(36,38),	(39,39),	(40,42),	(43,43),	(44,46),	(47,47),	(48,49),	(50,50),	(51,51),	(52,52),	(53,55),	(56,56),	(57,59),	(60,60),	(61,62),	(63,63),	(64,66),	(67,67),	(68,70),	(71,71),	(72,74),	(75,75),	(76,78),	(79,79),	(80,95),	(96,96),	(97,110),	(111,111),	(112,125),	(126,126),	(127,141)]
x= pd.read_fwf('SMKE192006',header=None,colspecs=col_specs,dtype=str)
x= pd.read_csv('SMKE192006',skiprows=1,header=None, dtype=str,sep='|')

cols = ['MAF-COMCODE',		'MAF-SITC',		'MAF-RECORD-TYPE',		'MAF-COD-SEQUENCE',		'MAF-COD-ALPHA',		'MAF-ACCOUNT-MMCCYY',		'MAF-PORT-SEQUENCE',		'MAF-PORT-ALPHA',		'MAF-FLAG-SEQUENCE',		'MAF-FLAG-ALPHA',		'MAF-TRADE-INDICATOR',		'MAF-CONTAINER',		'MAF-MODE-OF-TRANSPORT',		'MAF-INLAND-MOT',		'MAF-GOLO-SEQUENCE',		'MAF-GOLO-ALPHA',		'MAF-SUITE-INDICATOR',		'MAF-PROCEDURE-CODE',		'MAF-VALUE',		'MAF-QUANTITY-1',		'MAF-QUANTITY-2',		'MAF-INDUSTRIAL-PLANT-COMCODE']
x.columns = cols
x=x.dropna(axis=1, how='all')

###suppressions
x2=x[x['MAF-RECORD-TYPE']=='002']
x3=x[x['MAF-RECORD-TYPE']=='003']