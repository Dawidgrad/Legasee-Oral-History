# NER Evaluation with aligned text
# Need a subdirectory with NER outputs
# Need gold standard data in same level as code
import json
import re
import pandas as pd
from thefuzz import fuzz
from thefuzz import process
import os
import argparse
#!pip install thefuzz[speedup]


parser = argparse.ArgumentParser()
parser.add_argument("--outputs", type=str, help="dir with ner_outputs", default=None)
parser.add_argument("--gold", type=str, help="gold standard with ner annotation", default=None)
args = vars(parser.parse_args())



# Aggregating NE information into a dataframe
def tabulating(text, col):
        df = pd.DataFrame(columns=col)
        for i in range(len(patterns)):
                pattern = patterns[i]
                for a in re.finditer(pattern, text):
                        NE = a.group(i+2)
                        if i==1:
                                NE += a.group(5)
                        Label = 'Count'
                        col_len = len(col)
                        if NE in df.index:
                                df.loc[NE,Label] += 1
                        else:
                                idx = col.index(Label)
                                Labels = [0]*col_len
                                Labels[idx] = 1
                                df.loc[NE] = Labels
        return df


# Generates a dataframe for the intersection of two df
def intersection(df1, df2):
        df = pd.DataFrame(columns=col)
        for NE in df1.index:
                pseudo_match = process.extractOne(NE, df2.index)
                if pseudo_match != None:
                    if pseudo_match[1] >= 90:
                    #if NE in df2.index:
                            entry = NE if len(NE) < len(pseudo_match[0]) else pseudo_match[0]
                            df.loc[entry] = [min(df1.loc[NE,Label], df2.loc[pseudo_match[0],Label]) for Label in col]
        return df

# Evaluation based on absolute matches
def eval_crude(df_output, df_gold):
        df_inter = intersection(df_output, df_gold)
        if len(df_output.index) == 0:
          precision =0
        else:
          precision = len(df_inter.index)/len(df_output.index)
        if len(df_gold.index) == 0:
          recall =0
        else:
          recall = len(df_inter.index)/len(df_gold.index)
        if recall == 0 and precision ==0:
          F1 = 0
        else:
          F1 = 2*precision*recall/(precision+recall)
        return precision, recall, F1

# Evaluation based on counts of matches depsite label
def eval_weight(df_output, df_gold):
        df_inter = intersection(df_output, df_gold)
        df_output["sum"] = df_output.sum(axis=1)
        df_gold["sum"] = df_gold.sum(axis=1)
        df_inter["sum"] = df_inter.sum(axis=1)

        if sum(df_output['sum']) == 0:
          precision =0
        else:
          precision = sum(df_inter['sum'])/sum(df_output['sum'])

        if sum(df_gold['sum']) == 0:
          recall = 0
        else:
          recall = sum(df_inter['sum'])/sum(df_gold['sum'])      
        
        if precision == 0 and recall == 0:
                F1 = 0
        else:
                F1 = 2*precision*recall/(precision+recall)
        return round(precision*100,2), round(recall*100,2), round(F1*100,2)





# Open relevant files
output_files = os.listdir(args['outputs'])
gold_file = args['gold']
with open(gold_file, 'r') as f:
        gold_text = f.read().replace('\n', ' ')

output_texts = []
for output_file in output_files:
        with open(args['outputs']+'/'+output_file, 'r') as f:
                output_texts.append((f.read().replace('\n', ' '), output_file))

# Patterns that we are searching in the labelled code
col = ['Count']
pattern1 = r'\<([\w ]+)\>([\w ]+)\<\\([\w ]+)\>'
pattern2 = r'<([\w ]+)><([\w ]+)>([\w ]+)<\\([\w ]+)>([\w ]+)<\\([\w ]+)>' # embedded tags
patterns  = (pattern1, pattern2)

pd_col = ['Systems', 'P no count', 'R no count', 'F1 no count','P with count', 'R with count', 'F1 with count']
results = pd.DataFrame(columns=pd_col)
# Loop through all NER system and outputs precision, recall and F1
for output_text, output_file in output_texts:
        df_output = tabulating(output_text, col)
        df_gold = tabulating(gold_text, col)
        df_inter = intersection(df_gold, df_output)

        # eval = eval_crude(df_output, df_gold) + eval_weight(df_output, df_gold)

        system = [output_file.replace('_tagged_transcript.txt', '')] + list(eval_crude(df_output, df_gold)) + list(eval_weight(df_output, df_gold))
        current = pd.DataFrame([system],columns=pd_col)
        results = pd.concat([results, current])

results.to_csv('NER_results_table', index=False)