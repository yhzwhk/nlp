import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForQuestionAnswering
from dataset import CMRC2018Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载模型和 tokenizer
model = BertForQuestionAnswering.from_pretrained("./cmrc_bert_qa")
tokenizer = BertTokenizerFast.from_pretrained("./cmrc_bert_qa")
model.to(device)
model.eval()

# 2. 加载测试数据（dev 集）
test_dataset = CMRC2018Dataset("data/cmrc2018_dev.json", tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 3. 计算 EM
em_list = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        start_preds = torch.argmax(outputs.start_logits, dim=1)
        end_preds = torch.argmax(outputs.end_logits, dim=1)

        em = (
            (start_preds == start_positions) &
            (end_preds == end_positions)
        ).float()

        em_list.extend(em.cpu().tolist())

print(f"\nTest EM: {np.mean(em_list):.4f}")
