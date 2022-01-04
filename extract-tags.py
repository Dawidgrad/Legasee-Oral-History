## Extracting tags from csv file

import re
import pandas as pd
import argparse
import json


def extrator(args):
    file = args.file
    output = args.output
    df = pd.read_csv(file)
    category = {}
    for tag in df['Tags']:
        if type(tag) == str:
            categories = re.findall(u'(\||^)((\w| |/|\(|\)|-|&|!|@|#|\$|%|\*|/|)+)>', tag)
            tags = re.findall(u'>((\w| |/|\(|\)|-|&|!|@|#|\$|%|\*|/|)+)($|\|)', tag)
            for tag in zip(categories,tags):
                if tag[0][1] in category.keys():
                    category[tag[0][1]] = category[tag[0][1]] + [tag[1][0]]
                else:
                    category[tag[0][1]] = [tag[1][0]]
    for key in category.keys():
        category[key] = list(set(category[key]))
        category[key].sort()

    category2 = {}
    for superkey in category.keys():
        value = category[superkey]
        tags = {}
        for tag in value:
            Flag = False
            for key in tags.keys():
                if key in tag and ' - ' in tag:
                    Flag = True
                    key_ref = key
            if Flag == False:
                if ' - ' in tag:
                    tags[tag.split(' - ')[0]]=[tag.split(' - ')[0], tag.split(' - ')[1]]
                else:
                    tags[tag]=[tag]
            else:
                tags[key_ref]= tags[key_ref] + [tag.split(' - ')[1]]
        category2[superkey]=tags

    #print(category2)
    with open(output, 'w') as f:
        json.dump(category2, f)
    print(f'--- File parsed and saves as: {output} ---')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='file name of csv file to parse', default='../Downloads/Veterans-Export-2021-September-27-2212 - Veterans-Export-2021-September-27-2212.csv')
    parser.add_argument('--output', type=str, help='Output name to save JSON file', default='tags.json')
    args = parser.parse_args()
    extrator(args)
