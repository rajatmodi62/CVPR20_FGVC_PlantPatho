import json

with open('./best_policy.json', 'r') as fid:
    a = json.loads(fid.read())
    print(len(a[0][0]))
