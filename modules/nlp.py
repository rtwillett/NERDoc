import numpy as np
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_lg")


class NERDocument:
    
    def __init__(self, text:str):
        
        self.nlp = spacy.load("en_core_web_lg")
    
        # text = text.upper()

        self.doc = self.nlp(text)
        
        self.extract_ner()

        self.ner_extracts = dict()
        for ent in self.entities.entity.unique():
            # print(ent)
            self.ner_extracts[ent] = self.summarize_entity_type(ent)

        # Combines names if single names (like last names) are tallied separately from full names
        if 'PERSON' in self.ner_extracts.keys():
            self.no_double_counts()
        
    def extract_ner(self):
        ner_tuplist = []
        for ent in self.doc.ents:
            ner_tuplist.append((ent.text, ent.label_))
            
        self.entities = pd.DataFrame(ner_tuplist, columns = ['name', 'entity'])
#         print('Hi')

    def summarize_entity_type(self, entity_type:str):
        df_counts = pd.DataFrame(self.entities.loc[self.entities.entity == entity_type].groupby(['name', 'entity'])['name'].count())
        # t.reset_index()
        df_counts.columns = ['counts']
        df_counts.reset_index()
        df_counts = df_counts.sort_values('counts', ascending=False)
        df_counts.reset_index(inplace=True)
        df_counts.reset_index(drop=True, inplace=True)
        return df_counts[['name', 'counts']]

    def no_double_counts(self):
    
        '''
        When a person in mentioned in a document, very often they are referred to elsewhere in a document by their first
        or last name (in media outlets usually by last name). This function takes the single names and counts them
        towards the matching longer name ('Zelensky' would could in the count for 'Vlodymyr Zelensky').
        '''

        # This piece of syntax is weird but I was troubleshooting something. 
        # I'll circle back to polish later
        df = self.ner_extracts['PERSON']
        
        # Find the single names (singletons)
        single_mask = np.array([True if len(n.split()) == 1 else False for n in df.name])
        singletons = df.loc[single_mask]
        
        # Subset the dataframe to only have data for the non-singletons 
        nonsingletons = df.loc[~single_mask]
        
        # For each singleton, check the non-singleton dataset and see if there is a match. 
        # If a match is found, the count for the singleton is added to the longer name.
        matched_singleton = []
        update_counts = []
        for s in list(zip(singletons.name, singletons.counts)):
        #     print(s[0] + '-'*10)
            for e in list(zip(nonsingletons.name, nonsingletons.counts)):
                if s[0] in e[0]:
        #             print(e[0])
                    matched_singleton.append(s[0])
                    update_counts.append((e[0], s[1] + e[1]))
        
        # Creates dataframe out of the matched and updated counts
        updated_names = pd.DataFrame(update_counts, columns = singletons.columns)
        
        # Find the names in the original dataframe that were not updated not matched singletons or matched from singletons.
        not_updated_names = list(set(df.name) - set(updated_names.name) - set(matched_singleton))
        
        # Subsets the original dataframe for the data that was not updated
        not_updated = df.loc[df.name.isin(not_updated_names)]
        
        # Return a concatenated dataframe of the updated records and non-updated records
        # return 
        self.ner_extracts['PERSON'] = pd.concat([updated_names, not_updated])