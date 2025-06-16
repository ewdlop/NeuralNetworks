# 📚 TextDataset 類詳解

`TextDataset` 是一個自定義的 PyTorch 數據集類，專門用於處理文本數據並為 Transformer 模型訓練做準備。

## 🏗️ 類結構概覽

```python
class TextDataset(Dataset):
    """文本數據集類"""
    def __init__(self, text_file, vocab_size=10000, max_seq_len=128)
    def preprocess_text(self, text)
    def build_vocab(self, vocab_size) 
    def tokenize_sentences(self)
    def __len__(self)
    def __getitem__(self, idx)
```

## 🔧 詳細功能分析

### 1. **初始化方法 `__init__`**

```python
def __init__(self, text_file, vocab_size=10000, max_seq_len=128):
    self.max_seq_len = max_seq_len
    
    # 讀取文本數據
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 清理和預處理文本
    self.sentences = self.preprocess_text(text)
    
    # 構建詞彙表
    self.build_vocab(vocab_size)
    
    # 將句子轉換為token序列
    self.tokenized_sentences = self.tokenize_sentences()
```

**作用**：
- 📁 讀取文本文件
- 🧹 清理和預處理文本
- 📖 構建詞彙表
- 🔢 將文本轉換為數字序列

**參數說明**：
- `text_file`: 輸入文本文件路徑
- `vocab_size`: 詞彙表大小（默認10,000）
- `max_seq_len`: 最大序列長度（默認128）

### 2. **文本預處理 `preprocess_text`**

```python
def preprocess_text(self, text):
    """預處理文本，分割成句子"""
    # 移除多餘空白和換行
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # 簡單的句子分割（以句號、問號、驚嘆號分割）
    sentences = re.split(r'[.!?]+', text)
    
    # 清理每個句子
    cleaned_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 10:  # 過濾太短的句子
            # 保留字母、數字、空格和基本標點
            sent = re.sub(r'[^a-zA-Z0-9\s\',;:-]', '', sent)
            sent = re.sub(r'\s+', ' ', sent).strip()
            if sent:
                cleaned_sentences.append(sent.lower())
    
    return cleaned_sentences[:5000]  # 限制數據量
```

**處理步驟**：
1. 🔄 **標準化空白**：將多個換行符和空格合併
2. ✂️ **句子分割**：根據標點符號 `.!?` 分割句子
3. 🧼 **清理文本**：移除特殊字符，保留基本標點
4. 📏 **過濾短句**：移除太短的句子（少於10個字符）
5. 🔤 **轉小寫**：統一文本格式
6. 📊 **限制數量**：只取前5000個句子（加快訓練）

**示例轉換**：
```
原始文本:
"First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak."

預處理後:
["before we proceed any further, hear me speak", "speak, speak"]
```

### 3. **詞彙表構建 `build_vocab`**

```python
def build_vocab(self, vocab_size):
    """構建詞彙表"""
    # 特殊token
    self.pad_idx = 0  # 填充token
    self.unk_idx = 1  # 未知token
    self.bos_idx = 2  # 句子開始token
    self.eos_idx = 3  # 句子結束token
    
    special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    
    # 統計詞頻
    word_counts = Counter()
    for sentence in self.sentences:
        words = sentence.split()
        word_counts.update(words)
    
    # 選擇最常見的詞
    most_common = word_counts.most_common(vocab_size - len(special_tokens))
    
    # 構建詞彙表
    vocab_list = special_tokens + [word for word, _ in most_common]
    self.vocab = {word: idx for idx, word in enumerate(vocab_list)}
    self.idx_to_vocab = {idx: word for word, idx in self.vocab.items()}
```

**詞彙表結構**：

| 索引 | Token  | 說明           |
|------|--------|----------------|
| 0    | `<PAD>` | 填充（序列對齊） |
| 1    | `<UNK>` | 未知詞         |
| 2    | `<BOS>` | 句子開始       |
| 3    | `<EOS>` | 句子結束       |
| 4    | the    | 最常見詞1      |
| 5    | and    | 最常見詞2      |
| ...  | ...    | ...           |

**特殊 Token 的作用**：
- **`<PAD>`**：填充較短的序列，使批次中所有序列長度相同
- **`<UNK>`**：表示詞彙表中不存在的詞
- **`<BOS>`**：標記句子開始，幫助模型理解序列起點
- **`<EOS>`**：標記句子結束，幫助模型知道何時停止生成

### 4. **Token化 `tokenize_sentences`**

```python
def tokenize_sentences(self):
    """將句子轉換為token序列"""
    tokenized = []
    for sentence in self.sentences:
        words = sentence.split()
        # 添加BOS和EOS標記
        tokens = [self.bos_idx] + [
            self.vocab.get(word, self.unk_idx) for word in words
        ] + [self.eos_idx]
        
        # 截斷或填充到固定長度
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens += [self.pad_idx] * (self.max_seq_len - len(tokens))
        
        tokenized.append(tokens)
    
    return tokenized
```

**轉換示例**：

```
步驟1 - 原句: "to be or not to be"
       ↓
步驟2 - 詞分割: ["to", "be", "or", "not", "to", "be"]
       ↓
步驟3 - 添加特殊token: [<BOS>, "to", "be", "or", "not", "to", "be", <EOS>]
       ↓
步驟4 - 轉換為索引: [2, 45, 67, 23, 89, 45, 67, 3]
       ↓
步驟5 - 填充到固定長度 (假設max_seq_len=12): 
        [2, 45, 67, 23, 89, 45, 67, 3, 0, 0, 0, 0]
```

### 5. **數據集接口方法**

```python
def __len__(self):
    return len(self.tokenized_sentences)

def __getitem__(self, idx):
    tokens = self.tokenized_sentences[idx]
    
    # 對於 Transformer 訓練，創建源序列和目標序列
    src = torch.tensor(tokens, dtype=torch.long)
    tgt_input = torch.tensor(tokens[:-1], dtype=torch.long)   # 不包含最後一個token
    tgt_output = torch.tensor(tokens[1:], dtype=torch.long)   # 不包含第一個token
    
    return src, tgt_input, tgt_output
```

**數據格式說明**：

```
原序列:    [<BOS>, w1, w2, w3, <EOS>, <PAD>, <PAD>, ...]
           
返回的三個張量:
src:       [<BOS>, w1, w2, w3, <EOS>, <PAD>, <PAD>, ...]  # 編碼器輸入（完整序列）
tgt_input: [<BOS>, w1, w2, w3, <EOS>, <PAD>, ...]        # 解碼器輸入（去掉最後一個）
tgt_output:[w1, w2, w3, <EOS>, <PAD>, <PAD>, ...]        # 解碼器目標（去掉第一個）
```

## 🎯 為什麼這樣設計？

### 1. **Teacher Forcing 訓練策略**

Teacher Forcing 是訓練序列生成模型的標準技術：

- 在訓練階段，解碼器能看到正確的前一個 token
- `tgt_input` 和 `tgt_output` 錯位一個位置
- 這加快了訓練收斂速度

**示例**：
```
要學習生成: "hello world"

tgt_input:  [<BOS>, hello]      # 解碼器輸入
tgt_output: [hello, world]      # 期望輸出

模型學習: 給定 <BOS> 輸出 "hello"，給定 "hello" 輸出 "world"
```

### 2. **固定序列長度的必要性**

```python
# 批次處理需要相同形狀的張量
batch = [
    [2, 45, 67, 3, 0, 0],    # 句子1（填充了2個<PAD>）
    [2, 12, 34, 56, 78, 3],  # 句子2（恰好填滿）
    [2, 23, 45, 3, 0, 0],    # 句子3（填充了2個<PAD>）
]
# 所有序列長度相同，可以組成批次進行並行處理
```

### 3. **Transformer 架構適配**

Encoder-Decoder Transformer 需要：
- **編碼器輸入** (`src`)：完整的源序列
- **解碼器輸入** (`tgt_input`)：目標序列（用於 teacher forcing）
- **解碼器目標** (`tgt_output`)：期望輸出（用於計算損失）

## 💡 使用示例

```python
# 創建數據集
dataset = TextDataset('data/input.txt', vocab_size=5000, max_seq_len=64)

# 查看數據集信息
print(f"句子數量: {len(dataset)}")
print(f"詞彙表大小: {dataset.vocab_size}")
print(f"特殊token索引: PAD={dataset.pad_idx}, UNK={dataset.unk_idx}")

# 獲取一個樣本
src, tgt_input, tgt_output = dataset[0]
print(f"源序列形狀: {src.shape}")           # torch.Size([64])
print(f"目標輸入形狀: {tgt_input.shape}")    # torch.Size([63])
print(f"目標輸出形狀: {tgt_output.shape}")   # torch.Size([63])

# 創建數據加載器
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 遍歷批次
for batch_idx, (src_batch, tgt_input_batch, tgt_output_batch) in enumerate(dataloader):
    print(f"批次 {batch_idx}:")
    print(f"  源序列批次形狀: {src_batch.shape}")        # torch.Size([32, 64])
    print(f"  目標輸入批次形狀: {tgt_input_batch.shape}")  # torch.Size([32, 63])
    print(f"  目標輸出批次形狀: {tgt_output_batch.shape}") # torch.Size([32, 63])
    break
```

## 🔍 深入理解：從文本到模型輸入

讓我們跟蹤一個完整的例子：

```python
# 1. 原始文本（input.txt 中的一段）
raw_text = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.
"""

# 2. 預處理後的句子
sentences = [
    "before we proceed any further, hear me speak",
    "speak, speak"
]

# 3. 構建詞彙表（簡化版）
vocab = {
    '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
    'before': 4, 'we': 5, 'proceed': 6, 'any': 7, 
    'further': 8, 'hear': 9, 'me': 10, 'speak': 11
}

# 4. Token化第一個句子
sentence = "before we proceed any further, hear me speak"
words = ["before", "we", "proceed", "any", "further", "hear", "me", "speak"]
tokens = [2] + [vocab[word] for word in words] + [3]  # [2, 4, 5, 6, 7, 8, 9, 10, 11, 3]

# 5. 填充到固定長度（假設 max_seq_len=12）
padded_tokens = [2, 4, 5, 6, 7, 8, 9, 10, 11, 3, 0, 0]

# 6. 生成訓練數據
src = [2, 4, 5, 6, 7, 8, 9, 10, 11, 3, 0, 0]      # 編碼器輸入
tgt_input = [2, 4, 5, 6, 7, 8, 9, 10, 11, 3, 0]   # 解碼器輸入
tgt_output = [4, 5, 6, 7, 8, 9, 10, 11, 3, 0, 0]  # 解碼器目標
```

## 🚀 性能優化建議

1. **調整 `vocab_size`**：
   - 太小：很多詞變成 `<UNK>`，信息丟失
   - 太大：模型參數增加，訓練慢

2. **調整 `max_seq_len`**：
   - 太短：長句子被截斷
   - 太長：內存使用增加，訓練慢

3. **數據預處理優化**：
   - 可以添加更複雜的文本清理
   - 考慮使用 BPE 或 SentencePiece tokenization

4. **批次大小調整**：
   ```python
   # 根據GPU內存調整批次大小
   dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
   ```

## 🔧 自定義擴展

您可以根據需要擴展 `TextDataset`：

```python
class AdvancedTextDataset(TextDataset):
    def __init__(self, text_file, vocab_size=10000, max_seq_len=128, 
                 use_bpe=False, min_freq=2):
        self.use_bpe = use_bpe
        self.min_freq = min_freq
        super().__init__(text_file, vocab_size, max_seq_len)
    
    def build_vocab(self, vocab_size):
        # 可以添加最小詞頻過濾
        # 可以整合 BPE tokenization
        pass
    
    def preprocess_text(self, text):
        # 可以添加更複雜的文本清理
        # 例如：處理特殊字符、標準化等
        pass
```

這個 `TextDataset` 類是整個 Transformer 訓練流程的基礎，它將原始文本轉換為模型可以理解和處理的數字格式！📚🤖 