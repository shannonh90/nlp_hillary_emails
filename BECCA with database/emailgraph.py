# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 11:37:59 2016

@author: Steve Trush, Shannon Hamilton, Shrestha Mohanty
"""
#All imports for this notebook
import nltk, re
from nltk.corpus import wordnet as wn
import string
import numpy as np
import os
from nltk.tag import StanfordNERTagger
from py2neo import authenticate, Graph, Node, Relationship

#EmailGraph                           
class EmailGraph:
    #http://py2neo.org/2.0/intro.html#nodes-relationships
    #Creates a New Graph (You will Need to Update this Function for your own install)
    def __init__(self, user, pwrd):
        authenticate("localhost:7474", user, pwrd)
        self.graph = Graph("http://localhost:7474/db/data/")
        java_path = "C:\ProgramData\Oracle\Java\javapath\java.exe"
        os.environ['JAVAHOME'] = java_path
        self.st = StanfordNERTagger('C:\stanford-ner-2015-12-09\classifiers\english.conll.4class.distsim.crf.ser.gz',\
                      'C:\stanford-ner-2015-12-09\stanford-ner.jar')                       
        self.stop_words = nltk.corpus.stopwords.words('english')
        self.legal_words = {"section","fw","re","ops","fyi","doc no","case no","subtitle","btw","usc","foia","chapter","u.s.c",\
               "report","attachment","attachments","note","amended", "ebook","subject","unclassified department of state case","doc",\
               "unclassified u.s. department of state","original message","project", "copyright", "pls", "you","u.s. department of state case no"}
    
    #process email: removes some of the headings before looking for keywords    
    def process_email(self, email):
        processed = ""
        for line in email.split('\n'):
            s = line.lower()
            if s.startswith("unclassified u.s. department of state") or \
               s.startswith("release in") or \
               s.startswith("original message") or \
               s.startswith("to:") or \
               s.startswith("from:") or \
               s.startswith("sent:") or  \
               s.startswith("cc:"):
                    pass
            else:
                if len(line) > 0 and line[-1] == '.':
                    processed = processed + line + ' '
                else:    
                    processed = processed + line + '. '
        return processed
        
    #filter_by_contents: receives a list of noun_phrases and filters out phrases contained in longer phrases elsewhere in the list
    def filter_by_contents(self,noun_phrases):
        in_others = []
        for i, candidate in enumerate(noun_phrases):
            for j, other in enumerate(noun_phrases):
                if i != j:
                    if candidate[0].lower() in other[0].lower() and candidate[0] != other[0]:    #compare each phrase with another
                        in_others.append(candidate)
        #filter out our identified 'duplicate' words and stopwords.
        filtered_words = [w for w in noun_phrases if w not in in_others and \
                          w[0].lower() not in self.legal_words and w[0].lower() not in self.stop_words]
    
        #create a Frequency Distribution
        unigram_fd = nltk.FreqDist(filtered_words)              
        #get the most common phrases
        common_noun_phrases = unigram_fd.most_common(20)
        
        result = []
        words = set([w[0][0].lower() for w in common_noun_phrases])
        for w in words:
            best_match = None
            for phrase in common_noun_phrases:
                if phrase[0][0].lower() == w:
                    if best_match is None:
                        best_match = phrase
                    else:
                        best_match = (best_match[0],best_match[1]+phrase[1])
            result.append(best_match)
        return sorted([w for w in result],key=lambda w: w[1],reverse=True)
    
    #filter_by_hypernym: receives a list of candidates and finds the best hypernym for each.
    #I started with code by Anna Swigart, ANLP 2015, and her concept of using a dictionary to store
    #terms from WordNet, however this code drastically departs from her algorithm.
    def filter_by_hypernym(self, candidates):
        #create a dictionary
        results = []
        for term in candidates: #loop through list of candidates
            synsets = wn.synsets(term[0][0],'n') #obtain the synsets for the phrase
            if len(synsets) >= 1:
                hypers = synsets[0].hypernyms() + synsets[0].instance_hypernyms()
                if len(hypers) >= 1:
                    results.append(((term[0][0],hypers[0].name().split('.')[0]),term[1]))
                else:
                    results.append(term)
            else:
                results.append(term)
        return results
    
    #algorithm for extracting key words from an email body
    def final_algorithm(self,email):
        #Create Sentences
        sentences = nltk.sent_tokenize((self.process_email(email)))
        tokenized_sentences = []
        for s in sentences:
            #get the tokens for each sentence that are filtered
            tokenized_sentences.append([word for word in nltk.word_tokenize(s) \
                                        if not re.search('[0-9]', word) and word.lower() not in self.legal_words and len(word) > 2])
        
        #separate the NER tagged entities from the rest 
        def get_entities(tags):
            result = []
            curr = []
            for ent in tags:
                if ent[1] == 'O':
                    if len(curr) > 0:
                        result.append(curr)
                        curr = []
                else:
                    if len(curr) > 0:
                        if not curr[0][1] == ent[1].lower():
                            result.append(curr)
                            curr = [(ent[0],ent[1].lower())]
                        else:
                            curr = curr + [(ent[0],ent[1].lower())]
                    else:
                        curr = [(ent[0],ent[1].lower())]
            return result
        
        #NER tag each of the sentences
        tagged_sents = self.st.tag_sents(tokenized_sentences)
        entity_names = [] 
        for s in tagged_sents:
            entity_names = entity_names + get_entities(s)
        
        #reorganize the entities for further processing
        def compress_entities(entities):
            new_list = []
            for entity in entities:
                result = " ".join([w[0] for w in entity])
                new_list.append((result,entity[0][1]))
            return new_list
        
        entity_names = compress_entities(entity_names)
        #print(entity_names)
    
        # Print unique entity names
        noun_phrases = entity_names    
    
        #Candidates Filtered by Duplicate Nouns and Rescored by Length
        noun_phrases = self.filter_by_contents(noun_phrases)
        #print(noun_phrases)
        
        
        #Candidate with better categories/hypernyms!
        noun_phrases = self.filter_by_hypernym(noun_phrases)
  
        #print("Email:\n" + email)
        print("Key Phrases:\n"+str(noun_phrases))

        return noun_phrases

    #clears out a graph
    def delete(self):
        self.graph.delete_all()

    #checks to see if a node exists in a graph
    #http://stackoverflow.com/questions/22134649/how-to-check-if-a-node-exists-in-neo4j-with-py2neo
    def find_existing(self, label, key, value):
        mynode = list(self.graph.find(label, property_key=key, property_value=value))
        # node found
        if len(mynode) > 0:
            return mynode[0]
        # no node found     
        else:
            return None
    
    #adds a new 'email' data element to the graph
    #code based on http://py2neo.org/2.0/intro.html#nodes-relationships
    def add_to_graph(self, data_element, terms):         
       
        #['Id', 'DocNumber', 'MetadataSubject', 'MetadataTo', 'Metadata From', 
        #'MetadataDateSent', 'ExtractedSubject', 'ExtractedTo',
        #'ExtractedFrom', 'ExtractedBodyText','RawText', 'Label']]
        email_id = data_element['DocNumber']
        email_feeling = data_element['NewLabel']
        email = self.find_existing("Email","docid",email_id)
        if email is None:
            if str(email_feeling) == '1':
                email_feelstr = 'emotional'
                n = 'E'
            else:
                email_feelstr = 'neutral'
                n = 'N'
            email = Node("Email", name = n, docid = email_id, tone=email_feelstr,\
                subject=data_element["ExtractedSubject"], date=data_element['MetadataDateSent'])
        s = email
        
        #add From nodes
        from_id_all = data_element['ExtractedFrom']
        if type(from_id_all) is str:
            for from_id_i in from_id_all.split(';'):
                from_id = from_id_i.strip().strip('\'')
                sender = self.find_existing("User","address",from_id)
                if sender is None:
                    sender = Node("User", address = from_id)
                s = s | Relationship(sender,"SENT", email)
        
        #add To nodes
        to_id_all = data_element['ExtractedTo']
        if type(to_id_all) is str:    
            for to_id_i in to_id_all.split(';'):
                to_id = to_id_i.strip().strip('\'')
                receiver = self.find_existing("User","address",to_id)
                if receiver is None:
                    receiver = Node("User", address = to_id)
                s = s | Relationship(receiver,"RECEIVED", email)
        
        #add Emotion nodes
        emote_all = data_element['Emotions']
        #print(emote_all)
        if type(emote_all) is str:
            print("Emotions: "+str(emote_all))
            for emote in emote_all.split(';'):
                if len(emote) > 0:
                    emotion = self.find_existing("Emotion","name",emote)
                    if emotion is None:
                        emotion = Node("Emotion", name = emote)
                    s = s | Relationship(email,"EMOTED", emotion)        
        
        self.graph.create(s)
        
        #add keywords and categories
        for item in range(0, len(terms)):
            keyword = terms[item][0][0]
            category = terms[item][0][1]
            n = self.find_existing("Keyword","name",keyword)
            if n is None:
                n = Node("Keyword",name=keyword)
            s = Relationship(email,"MENTIONS",n)
    
            c = self.find_existing("Category","name",category)
            if c is None:
                c = Node("Category",name=category)
            s = s | Relationship(n,"IS_TYPE_OF", c)
            self.graph.create(s)

    #get_random_emails - returns a number of random emails from a given data frame
    def get_random_emails(self, data_set, number):
        random_index = np.random.permutation(data_set.index)
        full_data_shuffled = data_set.ix[random_index,\
        ['Id', 'DocNumber', 'MetadataSubject', 'MetadataTo', 'Metadata From', 'MetadataDateSent',\
            'ExtractedSubject', 'ExtractedTo', 'ExtractedFrom','ExtractedBodyText','RawText',\
            'NewLabel', 'Emotions']]
        full_data_shuffled.reset_index(drop=True, inplace=True)
        #separate the training data from the development data
        return full_data_shuffled.loc[0:number-1]

    #adds a specified number of emails from a dataset
    def add_new_emails(self, num, total_df):
        selected_emails = self.get_random_emails(total_df, num)
        selected_emails["MetadataDateSent"].fillna(value='<blank>',inplace=True)
        selected_emails["ExtractedSubject"].fillna(value='<blank>',inplace=True)
        data_list = selected_emails["RawText"].values.tolist()
        subject_list = selected_emails["ExtractedSubject"].values.tolist()
        printable = set(string.printable)
        
        #for each email, extract the key words and then add to the graph
        for index in range(0, num):
            s = "".join(filter(lambda x: x in printable, data_list[index])) + ' . ' +\
                "".join(filter(lambda x: x != '<blank>' and x in printable, subject_list[index]))
            terms = self.final_algorithm(s)
            self.add_to_graph(selected_emails.loc[index],terms)

