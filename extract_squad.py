import json

# Load original big SQuAD file
with open("train-v2.0.json") as f:
    squad = json.load(f)

samples = []

# Pick 100 answerable samples
for article in squad["data"]:
    for para in article["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            if not qa["is_impossible"]:
                question = qa["question"]
                answer = qa["answers"][0]["text"]
                samples.append({
                    "context": context,
                    "question": question,
                    "answer": answer
                })
            if len(samples) >= 100:
                break
        if len(samples) >= 100:
            break
    if len(samples) >= 100:
        break

# Save smaller test file
with open("squad_sample.json", "w") as f:
    json.dump(samples, f, indent=2)

print("Saved 100 samples to squad_sample.json")
