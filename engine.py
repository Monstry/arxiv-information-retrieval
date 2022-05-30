from collections import defaultdict, Counter
import math
import copy
import preprocess
import json
from pprint import pprint
from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter
analyzer = RegexTokenizer() | LowercaseFilter() | StopFilter()


class SearchEngine(object):
    def __init__(self, papers_list, tokens_list, inverted_index):
        self.papers_list = papers_list
        self.tokens_list = tokens_list
        self.inverted_index = inverted_index
        self.papers_count = len(papers_list)
        self.avg_len = sum([t["length"]
                           for t in tokens_list]) / len(tokens_list)

    # Inverse Document Frequency
    def IDF(self, term):
        N = self.papers_count
        n = len(self.inverted_index.get(term, []))
        return math.log(((N - n + 0.5) / (n + 0.5))+1)

    # TF-IDF algorithm https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    def TF_IDF(self, query_terms, doc):
        score_sum = 0
        for term in query_terms:
            if term in doc["tokens_cnt"].keys():
                # Term Frequency
                TF = doc["tokens_cnt"].get(term, 0) / doc["length"]
                score = (1+math.log(TF))*self.IDF(term)
                score_sum += score
        return score_sum

    # BM25 algorithm https://en.wikipedia.org/wiki/Okapi_BM25
    def BM25(self, query_terms, doc, k1=1.5, b=0.75):
        score_sum = 0
        for term in query_terms:
            # Term Frequency
            TF = doc["tokens_cnt"].get(term, 0) / doc["length"]
            score = self.IDF(term) * (
                TF * (k1 + 1) / (TF + k1 *
                                (1 - b + b * doc["length"] / self.avg_len))
            )
            score_sum += score
        return score_sum

    # BM25+ algorithm https://en.wikipedia.org/wiki/Okapi_BM25
    def BM25_plus(self, query_terms, doc, k1=1.5, b=0.75, delta=1):
        score_sum = 0
        for term in query_terms:
            # Term Frequency
            TF = doc["tokens_cnt"].get(term, 0) / doc["length"]
            score = self.IDF(term) * (
                TF * (k1 + 1) / (TF + k1 *
                                 (1 - b + b * doc["length"] / self.avg_len)+delta)
            )
            score_sum += score
        return score_sum
    
    def get_doc_tfidf_dict(self, doc):
        doc_tfidf_dict = defaultdict(float)
        vector_len = 0
        for term in doc["tokens_cnt"].keys():
            doc_tfidf_dict[term] = self.TF_IDF([term], doc)
            vector_len += doc_tfidf_dict[term] * doc_tfidf_dict[term]
        vector_len = math.sqrt(vector_len)
        # divide vector length
        for term in doc["tokens_cnt"].keys():
            doc_tfidf_dict[term] = doc_tfidf_dict[term] / vector_len
        return doc_tfidf_dict

    def get_query_tfidf_dict(self, query_terms):
        cnt = Counter(query_terms)
        cnt_dict = dict(cnt)
        query_token = {
            "terms_seq": query_terms,
            "length": len(query_terms),
            "tokens_cnt": cnt_dict,
        }
        return self.get_doc_tfidf_dict(query_token)

    # Vector Space Model (VSM) https://en.wikipedia.org/wiki/Vector_space_model
    def vsm(self, query_terms, doc):
        doc_tfidf_dict = self.get_doc_tfidf_dict(doc)
        query_tfidf_dict = self.get_query_tfidf_dict(query_terms)
        score = 0
        for key in query_tfidf_dict.keys():
            score += (doc_tfidf_dict[key] * query_tfidf_dict[key])
        return score        

    # find all papers which contains query_terms
    def get_candidate_papers_ids(self, query_terms):
        candidates_papers_ids = []
        for term in set(query_terms):
            # [(paper_id, cnt),...]
            index_list = self.inverted_index.get(term, [])
            candidates_papers_ids.extend(item[0] for item in index_list)
        return set(candidates_papers_ids)  # remove redundancy using set

    def query(self, query_string: str, score_strategy:str):
        query_terms = [token.text for token in analyzer(query_string)]

        if query_terms == []:
            return []

        candidates_papers_ids = self.get_candidate_papers_ids(query_terms)

        rsp = []  # response
        for paper_id in candidates_papers_ids:
            doc = self.tokens_list[paper_id]
            if score_strategy == "tf_idf":
                score = self.TF_IDF(query_terms, doc)
            elif score_strategy == "BM25_plus":
                score = self.BM25_plus(query_terms, doc)
            elif score_strategy == "vsm":
                score = self.vsm(query_terms, doc)
            else:
                score = self.BM25(query_terms, doc)
            data = copy.deepcopy(self.papers_list[paper_id])
            data["score"] = score
            rsp.append(data)

        return rsp

    # rk_startegy: score(relevance), publish_time
    # desc_order: True, descending order; False, ascending order
    # areas: selected areas, empty represents all areas
    def filter(self, rsp, offset=0, limit=10, rk_startegy="score", desc_order=True, selected_areas=[]):

        if selected_areas:
            selected_areas = set(selected_areas)
            rsp = list(filter(
                lambda paper: len(
                    set(paper["categories"].split(",")) & selected_areas) > 0,
                rsp))

        if(rk_startegy == "publish_time"):
            rsp.sort(key=lambda x: x["published"], reverse=desc_order)
        else:
            rsp.sort(key=lambda x: x["score"], reverse=desc_order)

        return rsp[offset:offset+limit]


# initialize and return search engine
def get_search_engine():
    papers_list = preprocess.load_papers()
    with open("data/small/tokens_list.json", 'r') as f:
        tokens_list = json.load(f)
    with open("data/small/inverted_index.json", 'r') as f:
        inverted_index = json.load(f)

    return SearchEngine(papers_list, tokens_list, inverted_index)


if __name__ == "__main__":
    se = get_search_engine()

    rsp = se.query("RDMA")
    rsp = se.filter(rsp, rk_startegy="publish_time",
                    desc_order=False, selected_areas=["cs.NI"])

    pprint(rsp)
