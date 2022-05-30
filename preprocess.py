import csv
import json
from collections import Counter, defaultdict

# https://whoosh.readthedocs.io/en/latest/api/analysis.html
from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter
analyzer = RegexTokenizer() | LowercaseFilter() | StopFilter()


def load_papers(filename="data/small/papers.csv"):
    # list of all papers, each element is a dict
    papers_list = []
    with open(filename, "r") as f:
        f_csv = csv.DictReader(f)
        papers_list = list(f_csv)
    return papers_list


def tokenize(doc):
    doc_str = doc["title"] + " " + doc["summary"]
    # tokenize content string into terms sequence
    terms_seq = [token.text for token in analyzer(doc_str)]
    # count each token
    cnt = Counter(terms_seq)
    cnt_dict = dict(cnt)
    return {
        "terms_seq": terms_seq,
        "length": len(terms_seq),
        "tokens_cnt": cnt_dict,
    }


def get_tokens_list(papers_list):
    tokens_list = []

    for paper in papers_list:
        tokens_list.append(tokenize(paper))

    with open("data/small/tokens_list.json", "w") as f:
        f.write(json.dumps(tokens_list))

    return tokens_list


def build_inverted_index(tokens_list):

    inverted_index = defaultdict(list)
    for paper_id, tokens in enumerate(tokens_list):
        for term, cnt in tokens["tokens_cnt"].items():
            inverted_index[term].append((paper_id, cnt))

    with open("data/small/inverted_index.json", "w") as f:
        f.write(json.dumps(inverted_index))

    return inverted_index


def build_author_index(papers_list):
    author_index = defaultdict(list)
    for paper_id, paper in enumerate(papers_list):
        authors_list = paper["authors"].split(",")
        for author in authors_list:
            author_index[author].append(paper_id)

    with open("data/small/author_index.json", "w") as f:
        f.write(json.dumps(author_index))


def preprocess():

    papers_list = load_papers()

    tokens_list = get_tokens_list(papers_list)

    build_inverted_index(tokens_list)

    build_author_index(papers_list)


if __name__ == "__main__":
    preprocess()
    # papers_list = load_papers()
    # build_author_index(papers_list)
