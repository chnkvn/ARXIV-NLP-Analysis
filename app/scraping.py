import arxiv
import re
import requests
import srsly
from attrs import define, field, converters, asdict
from bs4 import BeautifulSoup
from typing import Iterable, Union, List, Set
from tqdm import tqdm

@define
class Paper_abstract:
    """ Create an instance of paper_abstract.
    A paper_abstract has:
    - a title(str)
    - some authors (List[str])
    - a raw abstract (str)
    - a publication date
    - an url (str)
    - one or several categories (List[str])
    - the processed abstract (str)"""
    title: str = field()
    authors: Union[List, Set] = field()
    raw_abstract:str = field()
    publication_date = field()
    paper_link:str = field()
    categories: Union[List, Set] = field(converter=list)
    abstract:str = field()
    @abstract.default
    def clean_abstract(self):
        """Replace latex formulas by "__FORMULA__"
        and remove latex formatting functions."""
        processed_abstract = ' '.join(self.raw_abstract.splitlines())
        processed_abstract = re.sub(r'\$[^\$]+\$', r'__FORMULA__', processed_abstract)
        processed_abstract = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', processed_abstract)
        return processed_abstract

def get_labels(url:str ='https://arxiv.org/' ) -> dict:
    """Get the name of categories and the tag of sub-categories.
    Return a dictionary of sub-category tag: category"""
    request = requests.get(url)
    soup = BeautifulSoup(request.content, 'lxml')
    content = soup.find('div', id='content')
    relevant_tags = content.find_all(['h2', 'strong'])
    category = None
    sub_cat = None
    sub_cat_to_cat = dict()
    for tag in relevant_tags:
      if '<h2>' in str(tag):
        category = tag.get_text()
      elif '</strong>' in str(tag) and category:
        sub_cat_to_cat[tag.get_text()] = category
    return sub_cat_to_cat

def fill_categories(map_labels_dict:dict, category_results: list, current_label:str ) -> list:
    """Assign one or several labels to an abstract.
    If a category result can be assigned to the abstract using map_label_dict, return the values
    that correspond to the relevant sub-categories.
    If at the end, no categories can be assigned to the abstract, assign the current label."""
    categories = {map_labels_dict[subcat.split('.')[0]]
                  for subcat in category_results
                  if subcat.split('.')[0] in map_labels_dict}
    return categories if len(categories)>0 else [current_label]

def get_abstracts(n_abstracts=100, save = False):
    """Get labels,
    then scrap n_abstracts (default:100) abstracts from arxiv per label."""
    client = arxiv.Client()
    subcat_to_label = get_labels()
    abstracts = []
    titles = set()
    # Extract main labels.
    for label in set(subcat_to_label.values()):
        print(f'{label} abstracts:')
        search = arxiv.Search(
        query = label,
        max_results = n_abstracts,
        sort_by = arxiv.SortCriterion.SubmittedDate
        )
        for r in tqdm(client.results(search)):
            # Scrap articles, add it if not already scrapped
            if r.title not in titles:
                titles.add(r.title)
                paper = Paper_abstract(title=r.title,
                                         authors = [author.name for author in r.authors],
                                         raw_abstract = r.summary,
                                         publication_date = r.published,
                                         paper_link =  r.pdf_url,
                                         categories = fill_categories(subcat_to_label, r.categories, label) )
                abstracts.append(paper)
        # Save into a jsonl file if wanted.
    if save:
        [a.abstract for a in abstracts]
        srsly.write_jsonl('data_models/abstracts.jsonl',
                          [asdict(abstract) for abstract in abstracts])
        print(f'Saved {len(abstracts)} articles')
    return abstracts

if __name__ == '__main__':
    get_abstracts(200, save=True)
