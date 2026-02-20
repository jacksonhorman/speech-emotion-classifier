import csv

csv_path = "src/ravdess_index.csv"
dataset = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset.append((row["path"], row["emotion"]))

print("Rows:", len(dataset))
print("First row:", dataset[0])
print("Last row:", dataset[-1])