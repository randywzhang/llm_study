from .data_loader import TransformerDataLoader

data_loader = TransformerDataLoader("tiny_shakespeare.txt")
data_loader.load_encoding("tiny_shakespeare_encoding.pkl")

xb, yb = data_loader.get_batch()

for b in range(data_loader.batch_size):
    for t in range(data_loader.block_size):
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"When input is {context.tolist()} the target: {target}")
