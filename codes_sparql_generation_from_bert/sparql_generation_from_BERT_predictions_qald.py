#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from collections import defaultdict
#from generator import decode


# In[2]:

def correct_relation_predicted_sparql(arg1, arg2, arg3, arg4, output):
	with open('index_to_url_map.json','r') as f_in:
    	index_to_url_map = json.load(f_in)
	index_to_url_map['61632']='r0'

	f=open(arg1,'r')
	pred=f.readlines()


	f=open(arg2)
	sparql=f.readlines()


	f=open(arg3)
	orig_test_data=f.readlines()


	f=open(arg4)
	bert_test_data=f.readlines()

	dict_replacement=defaultdict(dict)

	j=0
	for i in range(len(bert_test_data)):
    	words=bert_test_data[i].split(' [SEP] ')
    	question=words[0]
    	entity = '<http://dbpedia.org/resource/'+words[-1].split('\t')[0]+'>'
    	output = pred[i].strip('\n') 
    	if question==orig_test_data[j].strip('\n'):
        	dict_replacement[j][entity]='<'+index_to_url_map[output]+'>'
    	else:
        	j+=1
        	dict_replacement[j][entity]='<'+index_to_url_map[output]+'>'


	f1=open(output,'w')

	count=0
	for m in range(len(sparql)):
    	spql_words = sparql[m].split()
    	begin=spql_words.index('brack_open')
    	end=spql_words.index('brack_close')
    	begin_keywords=spql_words[:begin+1]
    	end_keywords=spql_words[end:]
    	triples=[x for x in spql_words[begin+1:end] if x!='sep_dot']
    	indices = [i for i, x in enumerate(spql_words) if x == "sep_dot"]
    #print(triples)
    	for i in range(len(triples)):
        	print(triples[i])
        	if triples[i] in dict_replacement[m].keys(): 
            	if i%3 == 0:
                	triples[i+1]=dict_replacement[m][triples[i]]
                #print(triples[i+1])
            	else:
                	triples[i-1]=dict_replacement[m][triples[i]]
                #print(triples[i+1])
    	query=begin_keywords+triples+end_keywords
    #print(triples)
    	for index in indices:
        	query.insert(index, 'sep_dot')
    	query= ' '.join (w for w in query)
    #print(query)
    	f1.write(query+'\n')
	f1.close()


# In[ ]:

import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument("--bert_pred", help='Provide the BERT prediction file path', type=str)
parser.add_argument("--sparql", help='Provide the path to CNN SPARQL prediction',type=str)
parser.add_argument("--test_en", help='Provide the path to Test Questions for CNN output',type=str)
parser.add_argument("--test_en_bert", help='Provide the path to Test Input Data for BERT Boost',type=str)
parser.add_argument("--output", help='Provide the file name to save the output',type=str)

args = parser.parse_args()
#file_name=os.path.join(args.file_path,'Translation_refined')
#content=open(file_name,'w')
order_files(args.bert_pred,args.sparql,args.test_en,args.test_en_bert,output)


