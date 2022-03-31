import pandas as pd


csv = pd.read_csv('results_all.csv')

rslts = []

for model in csv['Model'].unique():
    this_model = csv[csv['Model'] == model]
    for LMs in this_model['Language_Model'].unique():
        this_model_lm = this_model.loc[this_model['Language_Model'] == LMs]
        wer = round(this_model_lm['WER'].mean(), 3)
        min_wer = round(this_model_lm['WER'].min(), 3)
        max_wer = round(this_model_lm['WER'].max(), 3)
        std_wer = round(this_model_lm['WER'].std(), 3)
        cer = round(this_model_lm['CER'].mean(), 3)
        min_cer = round(this_model_lm['CER'].min(), 3)
        max_cer = round(this_model_lm['CER'].max(), 3)
        std_cer = round(this_model_lm['CER'].std(), 3)
        name = f"{model}_{LMs}"
        rslts.append([name, wer, min_wer, max_wer, std_wer, cer, min_cer, max_cer, std_cer])



for r in sorted(rslts, key=lambda x: x[1], reverse=True):
    print(f'{r[0]}: \nWER: {r[1]} (Min: {r[2]} - Max: {r[3]} -- SD: {r[4]}) \nCER: {r[5]} (Min: {r[6]} - Max: {r[7]} -- SD: {r[8]})\n')
