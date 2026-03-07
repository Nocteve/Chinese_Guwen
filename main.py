import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MT5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq
)
from torch.optim import AdamW
from tqdm import tqdm
from sacrebleu import corpus_bleu
from torch.amp import autocast, GradScaler

# ==================== 配置 ====================
DATA_DIR = './Classical-Modern/双语数据'
OUTPUT_DIR = './mt5_finetuned'
MODEL_NAME = 'google/mt5-small'
MAX_LEN = 128
BATCH_SIZE = 4
EPOCHS = 15
LR = 5e-5
WARMUP_STEPS = 500
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
NUM_WORKERS = 0

ACCUMULATION_STEPS = 4
EARLY_STOP_PATIENCE = 3
USE_AMP = True  # 混合精度开关
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')  # 检查点目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 自动选择混合精度类型
if USE_AMP and DEVICE == 'cuda':
    if torch.cuda.is_bf16_supported():
        AMP_DTYPE = torch.bfloat16
        print("使用 bfloat16 混合精度（稳定）")
    else:
        AMP_DTYPE = torch.float16
        print("警告：使用 float16 混合精度，可能不稳定。建议使用支持 bfloat16 的 GPU。")
else:
    AMP_DTYPE = None
    print("混合精度已禁用")

print(f'DEVICE: {DEVICE} USE_AMP: {USE_AMP}')

# ==================== 数据收集 ====================
def collect_sentence_pairs(data_dir):
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
                    print(f"警告：{root} 中行数不一致，跳过")
                    continue
                for src, tgt in zip(src_lines, tgt_lines):
                    if src and tgt:
                        pairs.append((src, tgt))
    print(f"收集到 {len(pairs)} 个句对")
    return pairs

# ==================== 数据集 ====================
class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src = "translate Classical Chinese to Modern Chinese: " + src

        model_inputs = self.tokenizer(
            src, max_length=self.max_len, truncation=True, padding=False
        )
        labels = self.tokenizer(
            text_target=tgt, max_length=self.max_len, truncation=True, padding=False
        )

        # 空序列保护
        if len(model_inputs['input_ids']) == 0:
            model_inputs['input_ids'] = [0]
            model_inputs['attention_mask'] = [0]
        if len(labels['input_ids']) == 0:
            labels['input_ids'] = [0]

        return {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'labels': labels['input_ids']
        }

# ==================== 训练工具函数 ====================
def _update_parameters(use_amp, scaler, optimizer, scheduler, model):
    """执行参数更新（梯度裁剪、优化器步进、调度器步进）"""
    if use_amp and scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

def train_epoch(model, loader, optimizer, scheduler, device,
                use_amp=False, amp_dtype=None, scaler=None, accumulation_steps=1):
    model.train()
    total_loss = 0
    step_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="Training")):
        batch = {k: v.to(device) for k, v in batch.items()}

        # 前向传播
        if use_amp and amp_dtype is not None:
            with autocast(device, dtype=amp_dtype):
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps
        else:
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps

        # NaN检测
        # if torch.isnan(loss):
        #     print(f"\n[NaN] Step {step+1}")
        #     raise ValueError("NaN loss")

        # 反向传播
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accumulation_steps
        step_loss += loss.item() * accumulation_steps

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1}: avg loss = {step_loss/200:.4f}")
            step_loss = 0.0

        if (step + 1) % accumulation_steps == 0:
            _update_parameters(use_amp, scaler, optimizer, scheduler, model)

    # 剩余批次
    if len(loader) % accumulation_steps != 0:
        _update_parameters(use_amp, scaler, optimizer, scheduler, model)

    return total_loss / len(loader)

def evaluate(model, loader, device, use_amp=False, amp_dtype=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_amp and amp_dtype is not None:
                with autocast(device, dtype=amp_dtype):
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            total_loss += outputs.loss.item()
    return total_loss / len(loader)

def validate_first_batch(model, loader, device, use_amp=False, amp_dtype=None):
    """验证第一个batch是否正常（无NaN）"""
    model.eval()
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        if use_amp and amp_dtype is not None:
            with autocast(device, dtype=amp_dtype):
                outputs = model(**batch)
        else:
            outputs = model(**batch)
    loss = outputs.loss
    print("\n=== 数据验证 ===")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"input_ids min/max: {batch['input_ids'].min().item()}/{batch['input_ids'].max().item()}")
    print(f"labels min/max: {batch['labels'].min().item()}/{batch['labels'].max().item()}")
    print(f"vocab size: {model.config.vocab_size}")
    print(f"logits mean/std: {outputs.logits.mean().item():.4f}/{outputs.logits.std().item():.4f}")
    print(f"loss: {loss}")
    if torch.isnan(loss):
        raise ValueError("第一个batch loss为NaN")
    print("数据验证通过")
    model.train()
    return loss

# ==================== 检查点 ====================
def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, patience_counter, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'config': model.config
    }
    path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch{epoch}.pt')
    torch.save(checkpoint, path)
    print(f"检查点已保存: {path}")
    if is_best:
        best_path = os.path.join(OUTPUT_DIR, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"最佳模型已保存: {best_path}")

def load_checkpoint(model, optimizer, scheduler, device):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_epoch')]
    if not checkpoints:
        return 1, float('inf'), 0  # 从epoch 1开始
    latest = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
    path = os.path.join(CHECKPOINT_DIR, latest)
    print(f"加载检查点: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    patience_counter = checkpoint['patience_counter']
    print(f"恢复训练: 从epoch {start_epoch} 开始")
    return start_epoch, best_val_loss, patience_counter

# ==================== 翻译/BLEU ====================
def translate_sentence(model, sentence, tokenizer, max_len=128, num_beams=4, device='cpu'):
    model.eval()
    src = "translate Classical Chinese to Modern Chinese: " + sentence
    inputs = tokenizer(src, max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_len, num_beams=num_beams,
                                 early_stopping=True, no_repeat_ngram_size=3)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compute_bleu(model, test_pairs, tokenizer, max_len=128, device='cpu', num_samples=1000):
    sample_pairs = random.sample(test_pairs, min(num_samples, len(test_pairs)))
    hypotheses, references = [], []
    for src, ref in tqdm(sample_pairs, desc="Translating test samples"):
        hyp = translate_sentence(model, src, tokenizer, max_len, device=device)
        hypotheses.append(hyp)
        references.append([ref])
    return corpus_bleu(hypotheses, references).score

# ==================== 主程序 ====================
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    random.seed(SEED)
    torch.manual_seed(SEED)

    # 加载模型和分词器
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    # 准备数据
    all_pairs = collect_sentence_pairs(DATA_DIR)
    random.shuffle(all_pairs)
    train_pairs = all_pairs[:int(0.9*len(all_pairs))]
    val_pairs = all_pairs[int(0.9*len(all_pairs)):int(0.95*len(all_pairs))]
    test_pairs = all_pairs[int(0.95*len(all_pairs)):]
    print(f"训练集: {len(train_pairs)} 验证集: {len(val_pairs)} 测试集: {len(test_pairs)}")

    # DataLoader
    train_dataset = TranslationDataset(train_pairs, tokenizer, MAX_LEN)
    val_dataset = TranslationDataset(val_pairs, tokenizer, MAX_LEN)
    test_dataset = TranslationDataset(test_pairs, tokenizer, MAX_LEN)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, max_length=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=data_collator, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=data_collator, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=data_collator, num_workers=NUM_WORKERS)

    # 数据验证
    print("\n验证第一个batch...")
    validate_first_batch(model, train_loader, DEVICE, use_amp=USE_AMP, amp_dtype=AMP_DTYPE)

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=LR)
    total_updates = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    warmup_steps = min(WARMUP_STEPS, total_updates)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_updates)

    # 混合精度 scaler
    scaler = GradScaler(DEVICE) if (USE_AMP and AMP_DTYPE == torch.float16) else None

    # 尝试加载检查点
    start_epoch, best_val_loss, patience_counter = load_checkpoint(model, optimizer, scheduler, DEVICE)

    # 训练循环
    print("开始训练...")
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE,
            use_amp=USE_AMP, amp_dtype=AMP_DTYPE, scaler=scaler, accumulation_steps=ACCUMULATION_STEPS
        )
        print(f"训练损失: {train_loss:.4f}")

        val_loss = evaluate(model, val_loader, DEVICE, use_amp=USE_AMP, amp_dtype=AMP_DTYPE)
        print(f"验证损失: {val_loss:.4f}")

        # 保存检查点（每个epoch都保存）
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, patience_counter, is_best)

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"早停：连续 {EARLY_STOP_PATIENCE} 个epoch未改善")
            break

    # 测试
    print("\n加载最佳模型进行测试...")
    best_checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pt'), map_location=DEVICE)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    bleu_score = compute_bleu(model, test_pairs, tokenizer, MAX_LEN, device=DEVICE)
    print(f"测试集 BLEU: {bleu_score:.2f}")

    # 示例翻译
    print("\n示例翻译：")
    sample_src = "学而时习之，不亦说乎？"
    print(f"原文: {sample_src}")
    print(f"译文: {translate_sentence(model, sample_src, tokenizer, MAX_LEN, device=DEVICE)}")