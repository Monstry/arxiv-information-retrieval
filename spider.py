from time import sleep
import arxiv
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)

# 40 categories of Computer Science on https://arxiv.org/
areas = ["Artificial Intelligence", "Computation and Language", "Computational Complexity", "Computational Engineering, Finance, and Science", "Computational Geometry", "Computer Science and Game Theory", "Computer Vision and Pattern Recognition", "Computers and Society", "Cryptography and Security", "Data Structures and Algorithms", "Databases", "Digital Libraries", "Discrete Mathematics", "Distributed, Parallel, and Cluster Computing", "Emerging Technologies", "Formal Languages and Automata Theory", "General Literature","Graphics", "Hardware Architecture", "Human-Computer Interaction", "Information Retrieval", "Information Theory", "Logic in Computer Science", "Machine Learning", "Mathematical Software", "Multiagent Systems", "Multimedia", "Networking and Internet Architecture", "Neural and Evolutionary Computing", "Numerical Analysis", "Operating Systems", "Other Computer Science", "Performance", "Programming Languages", "Robotics", "Social and Information Networks", "Software Engineering", "Sound", "Symbolic Computation", "Systems and Control"]

# axiv Result Object
#   entry_id: str,
#   updated: datetime = _DEFAULT_TIME,
#   published: datetime = _DEFAULT_TIME,
#   title: str = "",
#   authors: List['Result.Author'] = [],
#   summary: str = "",
#   comment: str = "",
#   journal_ref: str = "",
#   doi: str = "",
#   primary_category: str = "",
#   categories: List[str] = [],
#   links: List['Result.Link'] = [],
#   _raw: feedparser.FeedParserDict = None,


big_slow_client = arxiv.Client(
  page_size = 1000,
  delay_seconds = 60,
  num_retries = 10
)

for search_key in areas:
    
    search = arxiv.Search(
        query=search_key,
        max_results=10000,
        sort_by=arxiv.SortCriterion.Relevance
    )

    # collections
    entry_id_col = []
    published_col = []
    title_col = []
    authors_col = []
    summary_col = []
    primary_category_col = []
    categories_col = []
    pdf_url_col = []

    for result in big_slow_client.results(search):
        entry_id_col.append(result.entry_id)
        published_col.append(result.published)
        title_col.append(result.title)
        authors_col.append(
            ",".join([author.name for author in result.authors]))
        summary_col.append(result.summary)
        primary_category_col.append(result.primary_category)
        categories_col.append(",".join(result.categories))
        pdf_url_col.append(result.pdf_url)

    df = pd.DataFrame({
        "entry_id": entry_id_col,
        "published": published_col,
        "title": title_col,
        "authors": authors_col,
        "summary": summary_col,
        "primary_category": primary_category_col,
        "categories": categories_col,
        "pdf_url": pdf_url_col,
    })

    df.to_csv("papers/{}.csv".format(search_key), encoding='utf-8', index=False)
    
    print("Area: "+search_key+" DONE.")
    
# replace '\n' in the summary
def f(summary):
    return summary.replace('\n', ' ')

frames = []
for filename in os.listdir("papers"):
    frames.append(pd.read_csv("papers/"+filename))
df_all = pd.concat(frames)
df_all.drop_duplicates() # remove duplicated papers
df_all["summary"] = df["summary"].apply(f)
df_all.to_csv("data/large/papers.csv", encoding='utf-8', index=False)