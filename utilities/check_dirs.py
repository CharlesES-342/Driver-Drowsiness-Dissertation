'''
Used to check directories exist and that file locaitons are corrrect.
Following issues pointing to locations, this helped set a baseline to find a file from a simple
function call.
'''
from pathlib import Path

root = Path(r"C:/Users/smelt/Git_Repo_Dest/Driver-Drowsiness-Dissertation/TrainingData")

print("TrainingData folder contents:")
for p in root.iterdir():
    print("-", p.name)
