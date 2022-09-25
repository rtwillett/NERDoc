import numpy as np
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_lg")


class NERDocument:
    
    def __init__(self, text:str, nlp = nlp):
        
        # self.nlp = spacy.load("en_core_web_lg")
    
        # text = text.upper()

        self.nlp = nlp

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

def autosummarize(text, summary_ratio = 0.1, keyword_list = [], ban_list = []):
    import numpy as np
    import pandas as pd
    import re
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    from gensim.models import Word2Vec
    from scipy import spatial
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm.auto import tqdm

    sentences = sent_tokenize(text) 
    
    print("Processing text ....\n")
    
    # Takes only sentences that are not in table of contents (has '....' in the line) or are too short to be useful
    # Reducing the list and removing garbage improves performance because it reduces the matrix space of the vector comparisons
    sentences = [s.strip() for s in sentences if ('....' not in s) & (len(s) > 30) & ('___' not in s) & ('====' not in s)]
    
    # Makes all text lowercase, strips whitespace and English stop words
    sentences_clean=[re.sub(r'[^\w\s]','',sentence.lower()).strip() for sentence in sentences]
    stop_words = stopwords.words('english')
    sentence_tokens=[[words.strip() for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]
    
    # Instantiates the Word2Vec class object
    w2v=Word2Vec(sentence_tokens, vector_size=1, window=5, min_count=1, workers=4) 
    
    # Removes any None type sentences from the list
    sentence_tokens = [w if len(w) !=0 else [''] for w in sentence_tokens]
    
    # Extracts the word vector for each word in each sentence
    # Produces a list of lists
    # For the top level of the list the elements correspond to the sentence (list of word embeddings for the words in that sentence)
    sentence_embeddings = [w2v.wv[w] for w in sentence_tokens]
    
    # Goes through the list of sentences and calculates the mean of word embeddings in each sentence).
    # This produces a simple list of vectors corresponding to the "relative meaning" of each sentence. 
    sentence_embeddings_means = np.array([e.mean() for e in sentence_embeddings])
    
    # Creates an empty matrix of the dimensions for comparison of each sentence embedding
    similarity_matrix = np.zeros([len(sentence_embeddings_means), len(sentence_embeddings_means)])

    print("Evaluating sentences similarities ....")
    print("(This may take a few minutes)\n")
    
    # Calculates the cosine similarity of each possible sentence pair in the list
    # This is estimating the similarity of the meaning of each sentence compared to each other.
    # Values are put into the similarity matrix.
    with tqdm(total = len(sentences), desc = 'Processing sentences') as pbar:
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = cosine_similarity(sentence_embeddings_means[i].reshape(-1,1), sentence_embeddings_means[j].reshape(-1,1))
            pbar.update(1)

    print("Analyzing similarity matrix ....\n")

    # Takes the similarity matrix and uses networkx to calculate the relative importance of each sentence using the PageRank algorithm
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank_numpy(nx_graph)
    
    # Assigns the PageRank score to the sentence it belongs to. 
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    
    # Extract top ranked sentences as the summary. 
    top_sums = []
    if int(len(ranked_sentences)*summary_ratio) >=5:
        n_sentences = int(len(ranked_sentences)*summary_ratio)
    elif (len(ranked_sentences) < 5):
        n_sentences = len(ranked_sentences)
    else:
        n_sentences = 5            

    for i in range(n_sentences):
        top_sums.append(ranked_sentences[i][1])   
    
    # Putting the sentences back in order as they appear in the document.
    pattern_money = r'(\$\s?\d+\,?\d{0,}\,?\d{0,}\,?\d{0,}\,?\d{0,}\,?\.?\d{0,})'
    summary_ordered = [s.replace('\n', '') for s in sentences if (s in top_sums) | (re.search(pattern_money, s) is not None)]
    
    return summary_ordered

class ExtractiveSummarize():
    
    def __init__(self, text_string): 

        # from tika import parser
        # import tika
        # tika.initVM()

        from nltk.tokenize import sent_tokenize

        self.default_antikeywords = []
        
        self.content_text = text_string
        self.text = self.content_text # Placeholder for cleaning
        
        self.sentences = sent_tokenize(self.text)
    
    def autosummarize(self, summary_ratio = 0.1, keywords = [], antikeywords = [], money_priority = False):
        import numpy as np
        import pandas as pd
        import re
        import nltk
        from nltk.tokenize import sent_tokenize
        from nltk.corpus import stopwords
        from gensim.models import Word2Vec
        from scipy import spatial
        import networkx as nx
        from sklearn.metrics.pairwise import cosine_similarity
        from tqdm.auto import tqdm

        # # Takes only sentences that are not in table of contents (has '....' in the line) or are too short to be useful
        # # Reducing the list and removing garbage improves performance because it reduces the matrix space of the vector comparisons
        # sentences = [s.strip() for s in self.sentences if ('....' not in s) & (len(s) > 30) & ('___' not in s) & ('====' not in s)]

        # Makes all text lowercase, strips whitespace and English stop words
        sentences_clean=[re.sub(r'[^\w\s]',' ',sentence.lower()).strip() for sentence in self.sentences]
        stop_words = stopwords.words('english')
        self.sentence_tokens=[[words.strip() for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]

        # Instantiates the Word2Vec class object
        w2v=Word2Vec(self.sentence_tokens, vector_size=1, window=5, min_count=1, workers=4) 

        # Removes any None type sentences from the list
        self.sentence_tokens = [w if len(w) !=0 else [''] for w in self.sentence_tokens]

        # Extracts the word vector for each word in each sentence
        # Produces a list of lists
        # For the top level of the list the elements correspond to the sentence (list of word embeddings for the words in that sentence)
        self.sentence_embeddings = [w2v.wv[w] for w in self.sentence_tokens]

        # Goes through the list of sentences and calculates the mean of word embeddings in each sentence).
        # This produces a simple list of vectors corresponding to the "relative meaning" of each sentence. 
        self.sentence_embeddings_means = np.array([e.mean() for e in self.sentence_embeddings])

        # Creates an empty matrix of the dimensions for comparison of each sentence embedding
        self.similarity_matrix = np.zeros([len(self.sentence_embeddings_means), len(self.sentence_embeddings_means)])

        print("Evaluating sentences similarities ....")
        print("(This may take a few minutes)\n")

        # Calculates the cosine similarity of each possible sentence pair in the list
        # This is estimating the similarity of the meaning of each sentence compared to each other.
        # Values are put into the similarity matrix.
        with tqdm(total = len(self.sentences), desc = 'Processing sentences') as pbar:
            for i in range(len(self.sentences)):
                for j in range(len(self.sentences)):
                    if i != j:
                        self.similarity_matrix[i][j] = cosine_similarity(self.sentence_embeddings_means[i].reshape(-1,1), self.sentence_embeddings_means[j].reshape(-1,1))
                pbar.update(1)

        print("Analyzing similarity matrix ....\n")

        # Takes the similarity matrix and uses networkx to calculate the relative importance of each sentence using the PageRank algorithm
        nx_graph = nx.from_numpy_array(self.similarity_matrix)
        scores = nx.pagerank_numpy(nx_graph)

        self.keyword_sentences = []
        for s in self.sentences:
            for kw in keywords: 
                if (kw in s):
                    self.keyword_sentences.append(s)
                else: 
                    pass

        self.antikeyword_sentences = []
        antikeywords.extend(self.default_antikeywords)
        for s in self.sentences:
            for akw in antikeywords: 
                if (akw in s):
                    self.antikeyword_sentences.append(s)
                else: 
                    pass

        # Putting the sentences back in order as they appear in the document.
        pattern_money = r'(\$\s?\d+\,?\d{0,}\,?\d{0,}\,?\d{0,}\,?\d{0,}\,?\.?\d{0,})'
        self.money_sentences = [s for s in self.sentences if (re.search(pattern_money, s) is not None)]

        # Assigns the PageRank score to the sentence it belongs to. 
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(self.sentences)), reverse=True)
        ranked_sentences = list(zip(*ranked_sentences))
        self.ranked_sentences = list(ranked_sentences[1])
        ranked_sentences_sub = list(set(self.ranked_sentences) - set(self.antikeyword_sentences))

        # Take the ranked sentences with antikeyword sentences removed and put them back in their rank order.
        self.ranked_sentences_filtered = [s for s in self.ranked_sentences if (s in ranked_sentences_sub)]
        
        # Extract top ranked sentences as the summary. 
        if int(len(self.sentences)*summary_ratio) >=5:
            self.n_sentences = int(len(self.sentences)*summary_ratio)
        elif (len(self.sentences) < 5):
            self.n_sentences = len(self.sentences)
        else:
            self.n_sentences = 5

        self.top_sums = self.ranked_sentences_filtered[:self.n_sentences]
   
        if money_priority:
            important_sentences = (set(self.top_sums) | set(self.money_sentences) | set(self.keyword_sentences))
        else: 
            important_sentences = (set(self.top_sums) | set(self.keyword_sentences))

        return [s.replace('\n', ' ') for s in self.sentences if (s in important_sentences)]


    def extract_summary(self, summary_ratio = 0.1, keywords = [], antikeywords = [], money_priority = False):

        sent_list = self.autosummarize(summary_ratio, keywords, antikeywords, money_priority)

        summary = ' '.join(sent_list)

        return summary