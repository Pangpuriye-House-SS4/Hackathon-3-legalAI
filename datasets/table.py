import pandas as pd

committee = pd.read_csv("committee.csv")
pattern = pd.read_csv("patterns.csv")
raw = pd.read_csv("raw.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
submit = pd.read_csv("sample_submission.csv")

# tokenizer 
rgno = [[str(first + " " + last) for first , last in zip(committee.loc[committee['rgno'] == rgo , 'fname'] , committee.loc[committee['rgno'] == rgo , 'lname'])] for rgo in train['rgno']]
patt_txt = [ pattern.loc[pattern['Pattern'] == int(p_txt) , 'Pattern.1'].values[0] for p_txt in train['pattern'] ]

new_train = train.copy()

new_train['new_rgno'] = rgno
new_train['pattern_text'] = patt_txt

new_train.to_csv("new_train.csv" , index=False)
