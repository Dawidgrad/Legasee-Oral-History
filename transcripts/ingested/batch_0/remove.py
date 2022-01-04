from os import listdir as ls
import re
from num2words import num2words

############################################################################################################################################################################################
# This code flags illegal strings within a transcript and asks the user (YOU!) for a replacement
# It used the num2words package to suggest an selection of possible replacements to reduce typing 
# (Options 1-5 for different number formations)
# option 6 simply removes illegal characters, and if you have previously manually added a replacement
# for a word option 7 will use this cached replacement
############################################################################################################################################################################################

inthisdir = ls('.')
target_dir = './cleaned/'
intargetdir = ls(target_dir)

inthisdir = [f for f in inthisdir if f not in intargetdir] # avoid already processed files

def read(file:str) -> str:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    return lines

valid = [' ','e','t','o','a','i','n','h','s','r','l','d','u','y','w','m','c','g','f','p','b','k',"'",'v','j','x','q','z']

cache = {}

def strip_non_valid(word):
    return ''.join([c for c in word if c in valid])

for file_ in inthisdir: 
    if file_[-3:] == 'txt':
        print('-'*100)
        print(file_)
        print('-'*100)
        text = " ".join(read(file_)).lower()
        # replace . ; ? ! , \n with a space
        text = re.sub(r'[\.,;?!-]|\n', ' ', text)  
        # remove … and ‘
        text = re.sub(r'[…]|‘', ' ', text)
        # replace ’ with '
        text = re.sub(r'’', "'", text)
        # remove double quotes
        text = re.sub(r'“|”', ' ', text)
        # remove all hyphens including - –
        text = re.sub(r'[-–]', ' ', text)
        # replace & with and
        text = re.sub(r'&', ' and ', text)
        # remove everything inside brackets, including brackets
        text = re.sub(r'\(.*?\)', ' ', text)
        # remove double spaces
        text = re.sub(r'\s+', ' ', text)
        new_text = []
        spltxt = text.split(' ')
        for i, word in enumerate(spltxt):
            if word.strip() == '':
                continue
            # any character that is not in the list of valid characters is flagged
            if not all([c in valid for c in word]):
                # retrieve 5 words before and after if not at the beginning or end
                print('-'*50)
                print(" ".join(spltxt[((i-5) if (i-5)>0 else 0):(i+5)]))
                justnums = re.sub(r'\D', '', word)
                justnums = justnums if len(justnums) > 0 else 0 # stop num2words from crashing
                default = re.sub(r'[^\w\s]', ' ', num2words(justnums))
                print(f'1. {word} -> {default} ?')
                ordinal = re.sub(r'[^\w\s]', ' ', num2words(justnums, ordinal=True))
                print(f'2. {word} -> {ordinal} ?')
                ordinal_num = re.sub(r'[^\w\s]', ' ', num2words(justnums, to='ordinal_num'))
                print(f'3. {word} -> {ordinal_num} ?')
                year = re.sub(r'[^\w\s]', ' ', num2words(justnums, to='year'))
                print(f'4. {word} -> {year} ?')
                currency = re.sub(r'[^\w\s]', ' ', num2words(justnums, to='currency'))
                print(f'5. {word} -> {currency} ?')

                takeout = ''.join(char for char in word if char in valid)
                print(f'6. {word} -> {takeout} ?')
                if word in cache:
                    print(f'7. {word} -> {cache[word]} ?')
                print('-'*50)
                print('Enter number to pick option or type manually')
                replace = input(f'Replace "{word}" with: ')
                print('-'*50)
                if replace.isdigit():
                    if int(replace) == 1:
                        new_text.append(strip_non_valid(default))
                    elif int(replace) == 2:
                        new_text.append(strip_non_valid(ordinal))
                    elif int(replace) == 3:
                        new_text.append(strip_non_valid(ordinal_num))
                    elif int(replace) == 4:
                        new_text.append(strip_non_valid(year))
                    elif int(replace) == 5:
                        new_text.append(strip_non_valid(currency))
                    elif int(replace) == 6:
                        new_text.append(strip_non_valid(takeout))
                    elif int(replace) == 7 and word in cache:
                        new_text.append(cache[word])
                else:
                    new_text.append(strip_non_valid(replace))
                    cache[word] = strip_non_valid(replace)
                print(new_text[-1])
            else:
                new_text.append(word)
            ###print(new_text[-1])
            
        text = " ".join(new_text)

        with open(target_dir + file_, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'{file_} -> {len(text)} characters')
