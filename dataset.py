import json
import torch
from torch.utils.data import Dataset

class CMRC2018Dataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(file_path, encoding="utf-8") as f:
            dataset = json.load(f)

        for article in dataset["data"]:
            for para in article["paragraphs"]:
                context = para["context"]

                for qa in para["qas"]:
                    question = qa["question"]

                    if not qa.get("answers"):
                        continue

                    ans = qa["answers"][0]
                    answer_text = ans["text"]
                    answer_start = ans["answer_start"]

                    self.data.append({
                        "context": context,
                        "question": question,
                        "answer": answer_text,
                        "answer_start": answer_start
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        encoding = self.tokenizer(
            example["question"],
            example["context"],
            truncation="only_second",
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offsets = encoding.pop("offset_mapping")[0]
        start_char = example["answer_start"]
        end_char = start_char + len(example["answer"])

        start_pos, end_pos = 0, 0
        for i, (s, e) in enumerate(offsets):
            if s <= start_char < e:
                start_pos = i
            if s < end_char <= e:
                end_pos = i
                break

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["start_positions"] = torch.tensor(start_pos)
        item["end_positions"] = torch.tensor(end_pos)

        return item
