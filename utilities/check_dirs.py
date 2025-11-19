from pathlib import Path

root = Path(r"C:/Users/smelt/Git_Repo_Dest/Driver-Drowsiness-Dissertation/TrainingData")

print("TrainingData folder contents:")
for p in root.iterdir():
    print("-", p.name)
