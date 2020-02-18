#!/usr/bin/env python
import re

scope_long = []
with open('scope_clean.txt') as t:
    for line in t:
        scope_long.append(line.strip())
t.close()

scope_long.sort(key=len, reverse=True)

category_map = {}
with open('topics.tsv') as t:
    for line in t:
        line = line.strip()
        scope = re.sub(r'\t.*', '', line)
        category = re.sub(r'.*\t', '', line)
        category_map[scope] = category

t.close()

scope = list(category_map.keys())
scope.sort(key=len, reverse=True)

for s in scope:
    matched = []
    for sl in scope_long:
        if re.match(s, sl):
            matched.append(sl)
            print(category_map[s] + "|" + s + "|" + sl)
    scope_long[:] = [item for item in scope_long if item not in matched]
