import sys
from notify import edits1,edits2

plate = sys.argv[1]
for p in edits1(plate):
    print(p)
