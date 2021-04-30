#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from generator import decode, encode
import json


# In[ ]:


def graph_traversal(ent_obj, ent_subj, sparql, content):
    
    with open(ent_obj) as f:
        ent_test_obj = json.load(f)
    with open(ent_subj) as f:
        ent_test_sub = json.load(f)
    f1=open(sparql)
    sparql = f1.readlines()
    f1.close()
    
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
            #print(triples[i])
            if triples[i].startswith('<http://dbpedia.org/resource'):
            #print('hello')
        #if triples[i] in dict_replacement[m].keys(): 
                if i%3 == 0:
                #if triples[i] in dict_replacement[m].keys():
                    rel = triples[i+1].strip('<>').split('/')[-1]
                    x= ent_test_sub[triples[i].strip('<>')]
                    list_x = [t.split('/')[-1] for t in x]
                    w = [i for i,d in enumerate(list_x) if d==rel]
                    if len(w)>1:
                        new_rel = 'http://dbpedia.org/ontology/'+rel
                        triples[i+1]='<'+new_rel+'>'
                    elif len(w)==1:
                        idx = list_x.index(rel)
                        new_rel = x[idx]
                        triples[i+1]='<'+new_rel+'>'
                
                #print(triples[i+1])
                else:
                    rel = triples[i-1].strip('<>').split('/')[-1]
                    x= ent_test_obj[triples[i].strip('<>')]
                    list_x = [t.split('/')[-1] for t in x]
                    w = [i for i,d in enumerate(list_x) if d==rel]
                    if len(w)>1:
                        new_rel = 'http://dbpedia.org/ontology/'+rel
                        triples[i-1]='<'+new_rel+'>'
                    elif len(w)==1:
                        idx = list_x.index(rel)
                        new_rel = x[idx]
                        triples[i-1]='<'+new_rel+'>'
        query=begin_keywords+triples+end_keywords
    #print(triples)
        for index in indices:
            query.insert(index, 'sep_dot')
        query= ' '.join (w for w in query)
    
    #print(query)
        content.write(query+'\n')
    #break
    content.close()
    return


# In[ ]:


import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument("--ent_subj", help='Relation dictionary for entities as subject', type=str)
parser.add_argument("--ent_obj", help='Relation dictionary for entities as object',type=str)
parser.add_argument("--bert_pred", help='SPARQL output after BERT correction',type=str)
#parser.add_argument("--test_en_bert", help='Provide the path to Test Input Data for BERT Boost',type=str)
#parser.add_argument("--output", help='Provide the file name to save the output',type=str)
parser.add_argument("--file_path", help='Provide the path to save the output',type=Path)

args = parser.parse_args()
file_name=os.path.join(args.file_path,'Translation_graph_traversed.txt')
content=open(file_name,'w')
graph_traversal(args.ent_subj,args.ent_obj,args.bert_pred,content)

