import requests
from bs4 import BeautifulSoup

class ScrapeWebpage:
    
    def __init__(self, url): 
        
        self.url = url
        
        self.get_webpage_contents()
    
    def get_webpage_contents(self):
        
        req = requests.get(self.url)
        bs = BeautifulSoup(req.text)
        p_elements = bs.find_all('p')
        plines = [p.text for p in p_elements]
        
        self.text = '\n'.join(plines)