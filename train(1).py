import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForQuestionAnswering
from torch.optim import AdamW
from dataset import CMRC2018Dataset
import numpy as np
from tqdm import tqdm

# =========================
# 1. 基本配置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

EPOCHS = 10
BATCH_SIZE = 8
LR = 3e-5

# =========================
# 2. tokenizer & dataset
# =========================
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

train_dataset = CMRC2018Dataset("data/cmrc2018_train.json", tokenizer)
dev_dataset = CMRC2018Dataset("data/cmrc2018_dev.json", tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# 3. model & optimizer
# =========================
model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)

# =========================
# 4. 评估函数（EM）
# =========================
def evaluate(model, dataloader):
    model.eval()
    em_list = []

    with torch.no_grad():
        for batch in dataloader:
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

    return np.mean(em_list)

# =========================
# 5. 训练循环（核心）
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch in progress:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    em_score = evaluate(model, dev_loader)

    print(f"\nEpoch {epoch + 1}")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Dev EM: {em_score:.4f}\n")

# =========================
# 6. 保存模型
# =========================
model.save_pretrained("./cmrc_bert_qa")
tokenizer.save_pretrained("./cmrc_bert_qa")
