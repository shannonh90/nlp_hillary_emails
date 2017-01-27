# -*- coding: utf-8 -*-
#imports - consolidation of joint work by Steve and Avi
import pandas as pd
import numpy as np
import string

#read_file is a helper function to get the '|' delimited CSV into a data frame
def read_file(filename):
    #get the file
    df = pd.read_csv(filename, low_memory=False, delimiter = ',')
    #filter the null data
    filtered_data = df["RawText"].notnull()
    df_narrative = df[filtered_data]
    return df_narrative

#reads in the training data into a panda - Steve 
#(code based on ANLP Notebook Intro to Pandas by Marti Hearst and Andrea Gagliano)
def read_training_data(filename):
    df_narrative = read_file(filename)
    #print the report on category breakdown, might need these counts later
#    print("Creating training data... category breakdown:")
#    sorted_product_counts = df_narrative.Category.value_counts(ascending=True)
#    print(sorted_product_counts)
    #sorted_product_counts.plot(kind='barh', figsize=(8,6), title="Categories");
    return df_narrative
    
#reads in the test file into a panda - Steve (code based on ANLP Notebook Intro to Pandas by Marti Hearst and Andrea Gagliano)
def read_test_data(filename):
    #get the file
    df = read_file(filename)
    return df

def get_number(prompt):
    user = input(prompt)                   #get input from keyboard
    answers = {'1','2','3'}
    while user not in answers:    #make sure something was entered
        user = input(prompt)   #else, prompt again
    return user
 
#breaks the panda into a training set and a dev set -Steve 
#(code based on ANLP Notebook Intro to Pandas by Marti Hearst and Andrea Gagliano)
def get_labelling_sets(full_data, percent):
    #randomize the indices
    #break down the counts for the shuffled data
    rows, columns = full_data.shape
    train_size = round(rows*(1 - percent))
    #separate the training data from the development data
    train_data = full_data.loc[train_size:]
    print("Loading emails "+str(train_size)+" to "+str(train_size*2))
    rows, colums = train_data.shape
    train_size = round(rows*.33)
    print("Dividing emails into batches of "+str(train_size))
    return train_data[:train_size], train_data[train_size:train_size*2], train_data[train_size*2:train_size*3]
    
def get_random_emails(data_set, number):
    random_index = np.random.permutation(data_set.index)
    full_data_shuffled = data_set.ix[random_index,\
    ['Id', 'DocNumber', 'MetadataSubject', 'MetadataTo', 'Metadata From', 'MetadataDateSent', 'ExtractedSubject', 'ExtractedTo', 'ExtractedFrom', 'ExtractedBodyText','RawText']]
    full_data_shuffled.reset_index(drop=True, inplace=True)
    print("Giving you "+str(number)+" of emails to label...")
    #separate the training data from the development data
    return full_data_shuffled.loc[0:number-1]

def get_category(prompt):
    user = input(prompt)                   #get input from keyboard
    good = {'0','1'}
    answers = {''}
    chk_prompt = "Are you sure? (Hit Enter again to confirm -or- N for No)"
    confirm = input(chk_prompt)
    while confirm not in answers or user not in good:    #make sure something was entered
        user = input(prompt)   #else, prompt again
        confirm = input(chk_prompt)
    return user

def label_categories(data_set, number):
    cats = ['0. neutral', '1. AWESOMENESS']

    results = []
    data_list = data_set["RawText"].values.tolist()
    for index in range(0, number):
        print(cats)
        printable = set(string.printable)
        s = "".join(filter(lambda x: x in printable, data_list[index]))
        print(s)
        mycat = get_category('Email #'+str(index+1)+' Enter category (an integer, please): ')
        print('You labeled it ' + cats[int(mycat)])
        print('\n')
        results.append(int(mycat))
        
    print(results)
    return results
  
#the code that calls the above functions - puts the data into a data frame
df = read_training_data("Emails.csv")
shrestha_set, shannon_set, steve_set = get_labelling_sets(df,.5)
user = get_number("Enter 1 for Shrestha, 2 for Shannon, 3 for Steve: ")
if user == '1':
    my_set = shrestha_set
if user == '2':
    my_set = shannon_set
if user == '3':
    my_set = steve_set

num = 233
selected_emails = get_random_emails(my_set, num)
print(selected_emails.shape)
results = label_categories(selected_emails, num)

selected_emails['Label'] = ""
print(selected_emails['Label'])
selected_emails['Label'] = results
print(selected_emails.head())
selected_emails.to_csv('email_out'+user+'set2.csv' , index = False)


