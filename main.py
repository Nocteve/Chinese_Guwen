import os
import random
import argparse
from tqdm import tqdm
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sacrebleu import corpus_bleu
import math

# -------------------- 参数配置 --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./Classical-Modern/双语数据',
                    help='双语数据根目录')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='模型和中间文件保存目录')
parser.add_argument('--vocab_size', type=int, default=32000,
                    help='词表大小')
parser.add_argument('--max_len', type=int, default=128,
                    help='句子最大长度')
parser.add_argument('--d_model', type=int, default=512,
                    help='Transformer 模型维度')
parser.add_argument('--nhead', type=int, default=8,
                    help='注意力头数')
parser.add_argument('--num_encoder_layers', type=int, default=6,
                    help='编码器层数')
parser.add_argument('--num_decoder_layers', type=int, default=6,
                    help='解码器层数')
parser.add_argument('--dim_feedforward', type=int, default=2048,
                    help='前馈网络维度')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout 比例')
parser.add_argument('--batch_size', type=int, default=32,
                    help='批次大小')
parser.add_argument('--epochs', type=int, default=10,
                    help='训练轮数')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='学习率')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='训练设备')
parser.add_argument('--seed', type=int, default=42,
                    help='随机种子')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 固定随机种子
random.seed(args.seed)
torch.manual_seed(args.seed)

# -------------------- 步骤1：收集所有平行句对（使用 source.txt 和 target.txt）--------------------
def collect_sentence_pairs(data_dir):
    """递归遍历所有子目录，从每个 source.txt 和 target.txt 中读取句对"""
    pairs = []
    for root, dirs, files in os.walk(data_dir):
        if "source.txt" in files and "target.txt" in files:
            src_path = os.path.join(root, "source.txt")
            tgt_path = os.path.join(root, "target.txt")
            with open(src_path, "r", encoding="utf-8") as f_src, \
                 open(tgt_path, "r", encoding="utf-8") as f_tgt:
                src_lines = [line.strip() for line in f_src if line.strip()]
                tgt_lines = [line.strip() for line in f_tgt if line.strip()]
                if len(src_lines) != len(tgt_lines):
                    print(f"警告：{root} 中 source.txt 和 target.txt 行数不一致，跳过该目录")
                    continue
                for src, tgt in zip(src_lines, tgt_lines):
                    if src and tgt:
                        pairs.append((src, tgt))
    print(f"收集到 {len(pairs)} 个句对")
    return pairs

all_pairs = collect_sentence_pairs(args.data_dir)

# 划分训练/验证/测试集
random.shuffle(all_pairs)
train_pairs = all_pairs[:int(0.9*len(all_pairs))]
val_pairs = all_pairs[int(0.9*len(all_pairs)):int(0.95*len(all_pairs))]
test_pairs = all_pairs[int(0.95*len(all_pairs)):]
print(f"训练集: {len(train_pairs)}, 验证集: {len(val_pairs)}, 测试集: {len(test_pairs)}")

# 保存为临时文件，用于训练 SentencePiece
def write_temp_files(pairs, src_file, tgt_file):
    with open(src_file, "w", encoding="utf-8") as f_src, \
         open(tgt_file, "w", encoding="utf-8") as f_tgt:
        for src, tgt in pairs:
            f_src.write(src + "\n")
            f_tgt.write(tgt + "\n")

temp_src = os.path.join(args.output_dir, "train.src")
temp_tgt = os.path.join(args.output_dir, "train.tgt")
write_temp_files(train_pairs, temp_src, temp_tgt)

# -------------------- 步骤2：训练 SentencePiece 模型 --------------------
spm_model_prefix = os.path.join(args.output_dir, "spm_model")
# 合并源和目标语料一起训练
os.system(f"cat {temp_src} {temp_tgt} > {args.output_dir}/train.all")

spm.SentencePieceTrainer.train(
    input=os.path.join(args.output_dir, "train.all"),
    model_prefix=spm_model_prefix,
    vocab_size=args.vocab_size,
    character_coverage=1.0,  # 覆盖所有字符（中文需要）
    model_type="bpe",
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    pad_piece="[PAD]", unk_piece="[UNK]", bos_piece="[BOS]", eos_piece="[EOS]"
)

# 加载词表
sp = spm.SentencePieceProcessor()
sp.load(f"{spm_model_prefix}.model")

vocab_size = sp.get_piece_size()
pad_id = sp.pad_id()
unk_id = sp.unk_id()
bos_id = sp.bos_id()
eos_id = sp.eos_id()
print(f"词表大小: {vocab_size}")

# -------------------- 步骤3：定义数据集类 --------------------
class TranslationDataset(Dataset):
    def __init__(self, pairs, sp, max_len):
        self.pairs = pairs
        self.sp = sp
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        # 编码并截断
        src_ids = self.sp.encode(src)[:self.max_len-2]
        tgt_ids = self.sp.encode(tgt)[:self.max_len-2]
        # 添加特殊标记
        src_ids = [bos_id] + src_ids + [eos_id]
        tgt_ids = [bos_id] + tgt_ids + [eos_id]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    """动态 padding 并生成掩码"""
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(seq) for seq in src_batch]
    tgt_lens = [len(seq) for seq in tgt_batch]
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)

    padded_src = torch.zeros(len(batch), max_src_len, dtype=torch.long)
    padded_tgt = torch.zeros(len(batch), max_tgt_len, dtype=torch.long)
    for i, seq in enumerate(src_batch):
        padded_src[i, :len(seq)] = seq
    for i, seq in enumerate(tgt_batch):
        padded_tgt[i, :len(seq)] = seq

    src_pad_mask = (padded_src == pad_id)  # (batch, src_len)
    tgt_pad_mask = (padded_tgt == pad_id)  # (batch, tgt_len)

    return padded_src, padded_tgt, src_pad_mask, tgt_pad_mask

# 创建 DataLoader
train_dataset = TranslationDataset(train_pairs, sp, args.max_len)
val_dataset = TranslationDataset(val_pairs, sp, args.max_len)
test_dataset = TranslationDataset(test_pairs, sp, args.max_len)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=4, pin_memory=True)

# -------------------- 步骤4：定义 Transformer 模型 --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_pad_mask=None, tgt_pad_mask=None):
        src_emb = self.pos_encoder(self.embedding(src))
        tgt_emb = self.pos_encoder(self.embedding(tgt))

        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        return self.fc_out(output)

def generate_square_subsequent_mask(sz):
    """生成因果掩码（上三角矩阵）"""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# 初始化模型
model = TransformerModel(
    vocab_size=vocab_size,
    d_model=args.d_model,
    nhead=args.nhead,
    num_encoder_layers=args.num_encoder_layers,
    num_decoder_layers=args.num_decoder_layers,
    dim_feedforward=args.dim_feedforward,
    dropout=args.dropout
).to(args.device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# -------------------- 步骤5：训练函数 --------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt, src_pad_mask, tgt_pad_mask in tqdm(loader, desc="Training"):
        src = src.to(device)
        tgt = tgt.to(device)
        src_pad_mask = src_pad_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        tgt_len = tgt_input.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_len).to(device)

        optimizer.zero_grad()
        logits = model(src, tgt_input,
                       src_pad_mask=src_pad_mask,
                       tgt_pad_mask=tgt_pad_mask[:, :-1],
                       tgt_mask=tgt_mask)

        loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt, src_pad_mask, tgt_pad_mask in tqdm(loader, desc="Evaluating"):
            src = src.to(device)
            tgt = tgt.to(device)
            src_pad_mask = src_pad_mask.to(device)
            tgt_pad_mask = tgt_pad_mask.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_len = tgt_input.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_len).to(device)

            logits = model(src, tgt_input,
                           src_pad_mask=src_pad_mask,
                           tgt_pad_mask=tgt_pad_mask[:, :-1],
                           tgt_mask=tgt_mask)

            loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

print("开始训练...")
best_val_loss = float('inf')
for epoch in range(1, args.epochs + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
    val_loss = evaluate(model, val_loader, criterion, args.device)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        print(f"  -> 保存最佳模型 (val_loss={val_loss:.4f})")

# -------------------- 步骤6：推理函数（贪心解码） --------------------
def translate_sentence(model, sentence, sp, max_len=50, device='cpu'):
    model.eval()
    src_ids = sp.encode(sentence)
    src_ids = [bos_id] + src_ids[:max_len-2] + [eos_id]
    src = torch.tensor([src_ids]).to(device)
    src_pad_mask = (src == pad_id)

    with torch.no_grad():
        src_emb = model.pos_encoder(model.embedding(src))
        memory = model.transformer.encoder(
            src_emb,
            mask=None,
            src_key_padding_mask=src_pad_mask
        )

        tgt_ids = [bos_id]
        for _ in range(max_len):
            tgt = torch.tensor([tgt_ids]).to(device)
            tgt_len = tgt.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_len).to(device)
            tgt_pad_mask = (tgt == pad_id)

            tgt_emb = model.pos_encoder(model.embedding(tgt))
            output = model.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask
            )
            logits = model.fc_out(output[:, -1, :])
            next_token = logits.argmax(-1).item()
            if next_token == eos_id:
                break
            tgt_ids.append(next_token)

    translation = sp.decode(tgt_ids[1:])
    return translation

# -------------------- 步骤7：测试集 BLEU 评估 --------------------
def compute_bleu(model, test_pairs, sp, device):
    hypotheses = []
    references = []
    for src, ref in tqdm(test_pairs, desc="Translating test set"):
        hyp = translate_sentence(model, src, sp, max_len=args.max_len, device=device)
        hypotheses.append(hyp)
        references.append([ref])
    bleu = corpus_bleu(hypotheses, references)
    return bleu.score

# 加载最佳模型
model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))
bleu_score = compute_bleu(model, test_pairs, sp, args.device)
print(f"测试集 BLEU: {bleu_score:.2f}")

# 示例翻译
print("\n示例翻译：")
sample_src = "学而时习之，不亦说乎？"
print(f"原文: {sample_src}")
print(f"译文: {translate_sentence(model, sample_src, sp, device=args.device)}")