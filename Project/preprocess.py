import re

def preprocess_property_type(s: str):
    s = s.lower()
    types = ["hdb", "condo", "house", "bungalow", "apartment", "landed"]
    for t in types:
        if t in s:
            return t
    return "other"

def preprocess_tenure(s: str):
    s = s.lower()
    match = re.match("(\d+)-year leasehold.*", s)
    if match and match.groups():
        yr = int(match.groups()[0])
        if yr <= 200:
            yr = 100
        elif yr <= 500:
            yr = 500
        else:
            yr = 1000
        return "tenure-{}".format(yr)
    elif "freehold" in s:
        return "freehold"
    return "other"
    