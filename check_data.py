from agents.AIBrain.ml.fast_loader import FastLoader

loader = FastLoader()
loader.index_data()

query = "SELECT DISTINCT filename FROM klines"
files = loader.con.execute(query).fetchall()
print(f"Indexed files: {len(files)}")

# Sample some to see what we have
for f in files[:10]:
    print(f"  - {f[0]}")
