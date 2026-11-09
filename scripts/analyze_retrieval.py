import pandas as pd

df = pd.read_csv("results/retrieval_results.csv")

queries = df["query"].unique()

top1_agreement = 0
top5_overlap = []

for query in queries:

    query_data = df[df["query"] == query]

    frame = query_data[
        query_data["method"] == "FRAME"
    ].sort_values("rank")

    teacher = query_data[
        query_data["method"] == "Teacher"
    ].sort_values("rank")

    frame_results = frame["retrieved_image"].tolist()
    teacher_results = teacher["retrieved_image"].tolist()

    if frame_results[0] == teacher_results[0]:
        top1_agreement += 1

    overlap = len(
        set(frame_results) & set(teacher_results)
    )

    top5_overlap.append(overlap / 5 * 100)

    print("\nQuery:", query)
    print("FRAME top 5:", frame_results)
    print("Teacher top 5:", teacher_results)
    print("Top-5 overlap:", overlap, "/ 5")

print("\n================================")
print("RETRIEVAL AGREEMENT ANALYSIS")
print("================================")

print("Total queries:", len(queries))

print(
    "Top-1 agreement:",
    round(top1_agreement / len(queries) * 100, 2),
    "%"
)

print(
    "Average Top-5 overlap:",
    round(sum(top5_overlap) / len(top5_overlap), 2),
    "%"
)
