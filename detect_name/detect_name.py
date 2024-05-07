import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from pythainlp.tokenize import word_tokenize # pip install pythainlp

df = pd.read_csv("/home/hpcnc/cloud/SuperAI/Hack-Legal-Action/data/train.csv")
MAX_LENGHT = 480
text_list= []
for i in df['context']:
    if i not in text_list:
        text_list.append(i)

def detect_name_and_group(List_text):

    name = "pythainlp/thainer-corpus-v2-base-model"

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForTokenClassification.from_pretrained(name)

    def fix_span_error(words,ner):
        _ner = []
        _ner = ner
        _new_tag=[]

        for i,j in zip(words,_ner):
            #print(i,j)
            i=tokenizer.decode(i)
            if i.isspace() and j.startswith("B-"):
                j="O"
            if i=='' or i=='<s>' or i=='</s>':
                continue
            if i=="<_>":
                i=" "
            _new_tag.append((i,j))

        return _new_tag

    Answer = []
    for sentence in tqdm(List_text):

        cut = word_tokenize(sentence.replace(" ", "<_>"))  #. บรรทัดนี้จะทำการตัดคำ และรีเพลสด้วย ช่องว่าง
        inputs = tokenizer(cut,is_split_into_words=True,return_tensors="pt")  #. ให้ตัว Thai NER ตัดคำอีกรอบนึง

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        if len(ids[0]) > 512:   #! แก้ Data ให้แบ่งประโยคยาวๆออกเป็น 2 ประโยค
            continue 
        
        outputs = model(ids, attention_mask=mask)

        logits = outputs[0]

        predictions = torch.argmax(logits, dim=2) 

        predicted_token_class = []
        for t in predictions[0]:
            predicted = model.config.id2label[t.item()]
            predicted_token_class.append(predicted)

        Answer.append(fix_span_error(inputs['input_ids'][0],predicted_token_class))

    def merge_person_names(tokens):   

        merged_tokens = []  
        current_name = ""  
        
        for token, entity_type in tokens:
            if entity_type == 'B-PERSON':
                if current_name:
                    merged_tokens.append((current_name.strip(), 'B-PERSON'))
                    current_name = ""
                current_name += token
            elif entity_type == 'I-PERSON':
                current_name += token
            elif current_name:
                merged_tokens.append((current_name.strip(), 'B-PERSON'))
                current_name = ""
                merged_tokens.append((token, entity_type))
            else:
                merged_tokens.append((token, entity_type))

        
        if current_name:
            merged_tokens.append((current_name.strip(), 'B-PERSON'))
        
        return merged_tokens

    merge_string = []

    for ner_list in Answer:
        merge_string.append(merge_person_names(ner_list))

    def extract_names(data):

        #+ This function will group name
        names ,counter  = [] ,0
        current_sublist = []

        for item in data:
            if item[1] == 'B-PERSON':
                current_sublist.append(item[0])
                counter = 0
            else:
                if item[0] == "ลง":
                    names.append(current_sublist)
                    current_sublist = []
                    continue
                
                counter += 1
                if counter > 3:   
                    names.append(current_sublist)
                    current_sublist = []
            
        if len(current_sublist) > 0:
            names.append(current_sublist)

        return names

    name_sublists = []

    for token in merge_string:
        x = extract_names(token)
        filtered_list = [sublist for sublist in x if sublist]
        name_sublists.append(filtered_list)
        # print(filtered_list)

    return filtered_list