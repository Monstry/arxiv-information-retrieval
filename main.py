from flask import Flask, request, jsonify
from flask_cors import CORS
from engine import get_search_engine
import json

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/graph.json")
def graph():
    with open("data/less-miserable.json", "r") as f:
        r = json.load(f)
    return jsonify(r)


@app.route("/search", methods=["GET", "POST", "PUT"])
def query():
    query_str = request.args.get("query")
    
    # score strategy: tf_idf, BM25, BM25+
    score_strategy = request.args.get("score_strategy") or "BM25"
    
    # filter settings
    try:
        offset = int(request.args.get("offset")) or 0
    except:
        offset = 0
    
    try:
        limit = int(request.args.get("limit")) or 10
    except:
        limit = 10
        
    rk_startegy = request.args.get("rk_startegy") or "score"

    try:
        desc_order = bool(request.args.get("desc_order")) or True
    except:
        desc_order = True
        
    try:
        selected_areas = list(request.args.get("selected_areas").split(',')) or []
    except:
        selected_areas = []
    
    # query
    rsp = se.query(query_str, score_strategy)
    rsp = se.filter(rsp, offset, limit, rk_startegy, desc_order, selected_areas)
    
    return jsonify(rsp)


if __name__ == "__main__":
    se = get_search_engine()
    app.run(host="0.0.0.0", port=8000)