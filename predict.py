import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizerFast.from_pretrained("./cmrc_bert_qa")
model = BertForQuestionAnswering.from_pretrained("./cmrc_bert_qa")
model.to(device)
model.eval()

context = "CMRC2018 是一个用于中文机器阅读理解的数据集。"
question = "CMRC2018 是做什么用的？"

inputs = tokenizer(
    question,
    context,
    truncation="only_second",
    max_length=512,
    return_tensors="pt"
)

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

start = torch.argmax(outputs.start_logits, dim=-1).item()
end = torch.argmax(outputs.end_logits, dim=-1).item()

if end < start:
    end = start

answer = tokenizer.decode(
    inputs["input_ids"][0][start:end + 1],
    skip_special_tokens=True
)

print("Question:", question)
print("Answer:", answer)
