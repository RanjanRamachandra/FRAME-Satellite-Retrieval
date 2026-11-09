exec(open("scripts/diagnose_retrieval_scores.py").read().split('print("SIMILARITY DIAGNOSTICS")')[0])

print("TOP-5 EMBEDDING DISTANCES")
values, indices = torch.topk(similarity[0], 5)

for rank, (score, index) in enumerate(zip(values, indices), 1):
    db_embedding = db[index]
    distance = torch.norm(q[0] - db_embedding).item()

    print(
        rank,
        database_names[index.item()],
        f"Cosine: {score.item():.8f}",
        f"L2 distance: {distance:.8f}"
    )
