import numpy as np
import pandas as pd
import networkx as nx
import re
import json
import matplotlib.pyplot as plt

class DocumentGraphProcessing:
    
    def __init__(self, filepath, source = None, sink = None, min_deg = None, max_deg = None):
        
        self.FILEPATH = filepath
        self.source_col = source
        self.sink_col = sink
        self.min_deg = min_deg
        self.max_deg = max_deg
        
        self.detect_filetype()
    
    def detect_filetype(self):
        
        self.file_ext = self.FILEPATH.split('.')[-1].lower()
        
        open_func = {
            'csv': self.read_csv,
            'xlsx': self.read_excel,
            'xls': self.read_excel,
            'json': self.read_json,
            'gephi': self.read_gephi,
            'gml': self.read_gml
        }
        
        self.df = open_func[self.file_ext]()
        
        if self.file_ext not in ['gml', 'gephi']:
            if (self.source_col is None) or (self.sink_col is None):
                print("Enter sink and source for data processing")
                
                self.g = None
            else:
                self.df_to_edgelist()
                self.build_graph()
        
        if self.g is not None:
            self.n_nodes = len(self.g.nodes)
        
        try:
            self.calculate_centrality()
        except: 
            pass
    
    def read_csv(self, sep = ","):
        return pd.read_csv(self.FILEPATH)
    
    def read_excel(self): 
        return pd.read_excel(self.FILEPATH)
    
    def read_json(self):
        with open(self.FILEPATH, 'r') as f: 
            d = f.read()
        
        return json.loads(d)
    
    def read_gephi(self):
        pass
    
    def read_gml(self):
        self.g = nx.read_gml(self.FILEPATH)
        
        self.EDGELIST = pd.DataFrame([tuple(i.replace(" {'value':", ",").replace('}', '').split(',')) for i in nx.generate_edgelist(self.g)], columns = ['source' ,'sink'])

    def df_to_edgelist(self): 
        # Add line to convert '' to na for dropping
        df_sub = self.df.dropna(subset = [self.source_col, self.sink_col])

        source = df_sub[self.source_col]
        sink = df_sub[self.sink_col]
        self.all_edges = [e for e in list(zip(source, sink))]
        self.all_nodes = [l for l in list(set(source) | set(sink))]

        self.EDGELIST = pd.DataFrame(self.all_edges, columns = [self.source_col, self.sink_col])


        source_node_cols = [self.source_col] + [col for col in self.df.columns if (col != self.source_col) and (col != self.sink_col)]
        sink_node_cols = [self.sink_col] + [col for col in self.df.columns if (col != self.source_col) and (col != self.sink_col)]

        nodelist_df = pd.concat([
            df_sub[source_node_cols].rename(columns = {'handle': 'node_name'}),
            df_sub[sink_node_cols].rename(columns = {'original_author': 'node_name'})
        ])

        self.NODELIST = nodelist_df.drop_duplicates(subset = ['node_name'])

    def build_graph(self): 
        self.g = nx.Graph()

        [self.g.add_node(n) for n in self.all_nodes]
        [self.g.add_edge(i[0], i[1]) for i in self.all_edges]
    
        return None
    
    def write_csv(self, filename, sep = ","):
        self.EDGELIST.to_csv(f'./{filename}_el.csv', index=None, sep = sep)
        self.NODELIST.to_csv(f'./{filename}_nl.csv', index=None, sep = sep)
    
    def write_excel(self): 
        self.EDGELIST.to_excel(f'./{filename}_el.csv', index=None)
        self.NODELIST.to_excel(f'./{filename}_nl.csv', index=None)
    
    def to_json(self):
        import json
        
        return json.load(self.FILEPATH)
    
    def write_gephi(self):
        pass 

    def write_gml(self): 
        pass
    
    def calculate_centrality(self):
        deg = nx.degree_centrality(self.g)
        closeness = nx.closeness_centrality(self.g)
        betweenness = nx.betweenness_centrality(self.g)
        eigen = nx.eigenvector_centrality(self.g)
        cen = pd.DataFrame([deg, closeness, betweenness, eigen]).transpose()
        cen.columns = ['degree', 'closeness', 'betweenness', 'eigenvector']
        cen['degree_nonnormal'] = cen.degree.apply(lambda x: int(round(x * self.n_nodes)))
        
        self.centrality = cen

    def filter_graph_degree(self): 

        if (self.min_deg is None) and (self.max_deg is None):
            df = self.centrality.reset_index(drop=False)
        elif (self.min_deg is not None) and (self.max_deg is None):
            df = self.centrality.loc[self.centrality.degree_nonnormal > self.min_deg].reset_index(drop=False)
        elif (self.min_deg is None) and (self.max_deg is not None):
            df = self.centrality.loc[self.centrality.degree_nonnormal < self.max_deg].reset_index(drop=False)
        elif (self.min_deg is not None) and (self.max_deg is not None):
            df = self.centrality.loc[(self.centrality.degree_nonnormal > self.min_deg) & (self.centrality.degree_nonnormal < self.max_deg)].reset_index(drop=False)
        else:
            # Put in proper error handling here
            print("Input not recognized")
        
        
        df.columns = ['node_name'] + list(df.columns)[1:]
        self.sub_g = nx.induced_subgraph(self.g, df.node_name)
    
    def draw(self, scope = 'full'):  
        
        if scope == 'full': 
            g = self.g
        elif scope == 'subgraph':
            self.filter_graph_degree()
            g = self.sub_g
        else: 
            pass

        import matplotlib.pyplot as plt
        # larger figure size
        plt.figure(3,figsize=(12,12)) 
        nx.draw_kamada_kawai(g)
    
    def save_graph(self, filename, scope = 'full'):

        self.draw(scope = scope)
        plt.savefig(f'./export/{filename}.png')