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
    
def parse_available_unit_types(s: str):
    x = s.split(',')
    mi, mx = 1000, -1000
    for i in range(len(x)):
        if 'studio' in x[i]:
            continue
        if 'br' in x[i]:
            x[i] = x[i].split(' ')[-2]
        try:
            mi = min(mi, int(x[i]))
            mx = max(mx, int(x[i]))
        except:
            print(x)
    return {'type': 'studio' if 'studio' in x[0] else 'other', 'min': mi, 'max': mx}

def filter_outlier_price_per_sqft(data):
    data['price_per_sqft'] = data['price'] / data['size_sqft']
    print("before filtering: ", data.price_per_sqft.describe())
    data_copy = data.copy()
    MIN, MAX = 100, 100000
    outliers = data_copy[(data_copy['price_per_sqft'] < MIN) | (data_copy['price_per_sqft'] > MAX)]
    data = data[(data['price_per_sqft'] >= MIN) & (data['price_per_sqft'] <= MAX)]
    print("after filtering: ", data.price_per_sqft.describe())
    data = data.drop(['price_per_sqft'], axis=1)
    return data, outliers