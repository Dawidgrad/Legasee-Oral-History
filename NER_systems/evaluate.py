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
parser.add_argument("--output_path", type=str, help="dir with NER system output in subfolders for each person", default=None)
parser.add_argument("--gold_path", type=str, help="dir with gold annotated transcripts", default=None)
parser.add_argument("--start", type=int, help="give range of threshold's start", default=None)
parser.add_argument("--stop", type=int, help="give range of threshold's stop", default=None)
parser.add_argument("--step", type=int, help="give range of threshold's step", default=None)

args = vars(parser.parse_args())



# Aggregating NE information into a dataframe
def tabulating(text, col):
  col = ['Count', 'Exact', 'prod']
  pattern1 = r'\<([\w ]+)\>([\w ]+)\<\\([\w ]+)\>'
  pattern2 = r'<([\w ]+)><([\w ]+)>([\w ]+)<\\([\w ]+)>([\w ]+)<\\([\w ]+)>' # embedded tags
  patterns  = (pattern1, pattern2)
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
def intersection(df1, df2, threshold):
        col = ['Count', 'Exact', 'prod']
        df = pd.DataFrame(columns=col)
        for NE in df1.index:
                pseudo_match = process.extractOne(NE, df2.index)
                if pseudo_match != None:
                    if pseudo_match[1] >= threshold:
                            entry = NE if len(NE) < len(pseudo_match[0]) else pseudo_match[0]
                            df.loc[entry,'Count'] = min(df1.loc[NE,'Count'], df2.loc[pseudo_match[0],'Count'])
                            df.loc[entry,'Exact'] = 1 if pseudo_match[0]==NE else 0
                            df.loc[entry, 'prod'] = df.loc[entry,'Count'] * df.loc[entry,'Exact']

        return df

# Gives relevant numbers needed to compute prec, recall, F1, ratio
def eval(df_output, df_gold, df_inter):
  gold_nc = len(df_gold.index)
  output_nc = len(df_output.index)
  inter_nc = len(df_inter.index)
  exact_nc = sum(df_inter['Exact'])
  matches_nc = len(df_inter)

  gold_wc = sum(df_gold['Count'])
  output_wc = sum(df_output['Count'])
  inter_wc = sum(df_inter['Count'])
  exact_wc = sum(df_inter['prod'])
  matches_wc = sum(df_inter['Count'])
  results = [gold_nc, output_nc, inter_nc, exact_nc, matches_nc, gold_wc, output_wc, inter_wc, exact_wc, matches_wc]
  return results

# output_files = directory with a person's ner system outputs
# gold_file = txt file with gold annotated text
def result_per_name(output_texts, gold_text):
  # Open relevant files for a specific person
  col = ['Count', 'Exact', 'prod']
  pd_col = ['Systems', 'gold_nc', 'output_nc', 'inter_nc', 'exact_nc', 'matches_nc', 'gold_wc', 'output_wc', 'inter_wc', 'exact_wc', 'matches_wc']
  results = pd.DataFrame(columns=pd_col)
  start, stop, step = args['start'], args['stop'], args['step']


# Loop through all NER system and outputs values needed for micro F1
  df_gold = tabulating(gold_text, col)
  for output_text, system_name in output_texts:
          df_output = tabulating(output_text, col)
          df_inter = []

          for threshold in range(start, stop, step):
                  df_inter.append((intersection(df_output, df_gold, threshold), threshold))
          
          systems = []
          for df, threshold in df_inter:
                  eval_data = eval(df_output, df_gold, df)
                  system = [system_name+'_'+str(threshold)] + eval_data
                  systems.append(system)
          current = pd.DataFrame(systems,columns=pd_col)
          results = pd.concat([results, current])
  
  return results.set_index('Systems')


def accumulated_table(gold_path, output_path):
  df_per_name = []
  for name in os.listdir(output_path):
    with open(gold_path+'/'+name+'.txt', 'r') as g:
      gold_text = g.read().replace('\n', ' ')
    
    output_texts = []
    for system in os.listdir(output_path+'/'+name):
      with open(output_path+'/'+name+'/'+system, 'r') as f:
        output_texts.append((f.read().replace('\n', ' '), system.replace('_tagged_transcript.txt', '_')))

    df_per_name.append((result_per_name(output_texts, gold_text), name))

  df_all = df_per_name[0][0]*0
  for df, name in df_per_name:
      df_all += df

  return df_all

def convert_to_metrics(table):
#['Systems', 'gold_nc', 'output_nc', 'inter_nc', 'exact_nc', 'matches_nc', 'gold_wc', 'output_wc', 'inter_wc', 'exact_wc', 'matches_wc']
  metrics = []
  indexes = []
  for index, row in table.iterrows():
    p_nc = row['inter_nc']/row['output_nc']
    r_nc = row['inter_nc'] / row['gold_nc']
    f1_nc = 2*p_nc*r_nc/(p_nc+r_nc)
    e_ratio_nc = row['exact_nc']/row['matches_nc']
    p_wc = row['inter_wc']/row['output_wc']
    r_wc = row['inter_wc'] / row['gold_wc']
    f1_wc = 2*p_wc*r_wc/(p_wc+r_wc)
    e_ratio_wc = row['exact_wc']/row['matches_wc']
    metrics.append([p_nc, r_nc, f1_nc, e_ratio_nc, p_wc, r_wc, f1_wc, e_ratio_wc])
    indexes.append(index)
  
  df_metrics = pd.DataFrame(metrics,
                            index = indexes,
                            columns = ['p_nc', 'r_nc', 'f1_nc', 'e_ratio_nc', 'p_wc', 'r_wc', 'f1_wc', 'e_ratio_wc'],
                            )
  df_metrics = df_metrics*100
  return df_metrics.round(2)

gold_path = args['gold_path']
output_path = args['output_path']
table = accumulated_table(gold_path, output_path)
metrics = convert_to_metrics(table)
print(metrics)
o_file_name = 'NER_eval_'+output_path.replace('output','').replace('/', '_')
metrics.to_csv(o_file_name)