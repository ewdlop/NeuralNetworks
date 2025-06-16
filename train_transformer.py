import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import pickle
import os
import getpass  # 添加安全輸入模組
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from encoder_decoder_transformer import Transformer

class TextDataset(Dataset):
    """文本數據集類"""
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
        
        print(f"數據集信息:")
        print(f"- 總句子數: {len(self.sentences)}")
        print(f"- 詞彙表大小: {len(self.vocab)}")
        print(f"- 特殊token: <PAD>={self.pad_idx}, <UNK>={self.unk_idx}, <BOS>={self.bos_idx}, <EOS>={self.eos_idx}")
    
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
                # 簡單的清理：保留字母、數字、空格和基本標點
                sent = re.sub(r'[^a-zA-Z0-9\s\',;:-]', '', sent)
                sent = re.sub(r'\s+', ' ', sent).strip()
                if sent:
                    cleaned_sentences.append(sent.lower())
        
        return cleaned_sentences[:5000]  # 限制數據量以加快訓練
    
    def build_vocab(self, vocab_size):
        """構建詞彙表"""
        # 特殊token
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3
        
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
    
    def __len__(self):
        return len(self.tokenized_sentences)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_sentences[idx]
        
        # 對於 Transformer 訓練，我們創建源序列和目標序列
        # 這裡我們使用 "復述" 任務：模型學習重複輸入句子
        src = torch.tensor(tokens, dtype=torch.long)
        tgt_input = torch.tensor(tokens[:-1], dtype=torch.long)  # 不包含最後一個token
        tgt_output = torch.tensor(tokens[1:], dtype=torch.long)  # 不包含第一個token
        
        return src, tgt_input, tgt_output

def train_model():
    """訓練模型"""
    # 設備設置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 超參數
    batch_size = 16
    max_seq_len = 64  # 減小序列長度以加快訓練
    vocab_size = 5000  # 減小詞彙表大小
    d_model = 256     # 模型維度
    n_heads = 8       # 注意力頭數
    n_layers = 4      # 層數（減少以加快訓練）
    d_ff = 1024       # 前饋網絡維度
    dropout = 0.1
    learning_rate = 1e-4
    num_epochs = 10
    
    # 創建數據集
    print("創建數據集...")
    dataset = TextDataset('data/input.txt', vocab_size=vocab_size, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 創建模型
    print("創建模型...")
    model = Transformer(
        src_vocab_size=len(dataset.vocab),
        tgt_vocab_size=len(dataset.vocab),
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_layers,
        n_decoder_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        pad_idx=dataset.pad_idx
    ).to(device)
    
    # 計算參數數量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型參數數量: {total_params:,}")
    
    # 優化器和損失函數
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    
    # 訓練循環
    model.train()
    train_losses = []
    
    print("開始訓練...")
    for epoch in range(num_epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') # 創建進度條
        for batch_idx, (src, tgt_input, tgt_output) in enumerate(pbar):
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            
            # 前向傳播
            optimizer.zero_grad()
            
            # 獲取模型輸出（注意：模型返回概率分佈）
            output_probs = model(src, tgt_input)  # (batch, seq_len, vocab_size)
            
            # 將概率轉換回logits以計算損失
            output_logits = torch.log(output_probs + 1e-8)  # 添加小數避免log(0)
            
            # 計算損失 
            loss = criterion(
                output_logits.reshape(-1, output_logits.size(-1)), 
                tgt_output.reshape(-1)
            )
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪, 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', # 損失
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}' # 學習率
            })
        
        # 計算平均損失
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}: 平均損失 = {avg_loss:.4f}')
        
        # 更新學習率
        scheduler.step()
        
        # 每3個epoch生成一些示例
        if (epoch + 1) % 3 == 0:
            print("\n生成示例:")
            generate_samples(model, dataset, device, num_samples=2)
            print()
    
    # 保存模型
    save_path = 'transformer_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': dataset.vocab,
        'idx_to_vocab': dataset.idx_to_vocab,
        'config': {
            'src_vocab_size': len(dataset.vocab),
            'tgt_vocab_size': len(dataset.vocab),
            'd_model': d_model,
            'n_heads': n_heads,
            'n_encoder_layers': n_layers,
            'n_decoder_layers': n_layers,
            'd_ff': d_ff,
            'dropout': dropout,
            'pad_idx': dataset.pad_idx
        }
    }, save_path)
    print(f"模型已保存到: {save_path}")
    
    # 繪製訓練損失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    return model, dataset

def generate_samples(model, dataset, device, num_samples=3):
    """生成文本示例"""
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            # 隨機選擇一個輸入句子
            idx = torch.randint(0, len(dataset), (1,))
            src, _, _ = dataset[idx.item()]
            src = src.unsqueeze(0).to(device)  # 添加batch維度
            
            # 生成文本
            generated = model.generate(src, max_len=32, 
                                     start_token=dataset.bos_idx, 
                                     end_token=dataset.eos_idx)
            
            # 轉換為文本
            src_text = tokens_to_text(src[0], dataset.idx_to_vocab)
            gen_text = tokens_to_text(generated[0], dataset.idx_to_vocab)
            
            print(f"示例 {i+1}:")
            print(f"輸入: {src_text}")
            print(f"生成: {gen_text}")
            print("-" * 50)
    
    model.train()

def tokens_to_text(tokens, idx_to_vocab):
    """將token序列轉換為文本"""
    words = []
    for token in tokens:
        word = idx_to_vocab.get(token.item(), '<UNK>')
        if word in ['<PAD>', '<BOS>', '<EOS>']:
            if word == '<EOS>':
                break
            continue
        words.append(word)
    return ' '.join(words)

def test_model():
    """測試已保存的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 載入模型
    checkpoint = torch.load('transformer_model.pth', map_location=device)
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    idx_to_vocab = checkpoint['idx_to_vocab']
    
    # 重建模型
    model = Transformer(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("模型載入成功！")
    print("生成一些文本示例:")
    
    # 創建簡單的測試輸入
    test_sentences = [
        "the king is dead",
        "what is your name", 
        "i love you"
    ]
    
    model.eval()
    with torch.no_grad():
        for sentence in test_sentences:
            # 簡單的tokenization
            words = sentence.lower().split()
            tokens = [vocab.get('<BOS>', 2)] + [vocab.get(word, vocab.get('<UNK>', 1)) for word in words] + [vocab.get('<EOS>', 3)]
            
            # 填充到合適長度
            max_len = 32
            if len(tokens) < max_len:
                tokens += [vocab.get('<PAD>', 0)] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
            
            src = torch.tensor(tokens).unsqueeze(0).to(device)
            
            # 生成
            generated = model.generate(src, max_len=32, 
                                     start_token=vocab.get('<BOS>', 2), 
                                     end_token=vocab.get('<EOS>', 3))
            
            # 轉換為文本
            gen_text = tokens_to_text(generated[0], idx_to_vocab)
            
            print(f"輸入: {sentence}")
            print(f"生成: {gen_text}")
            print("-" * 40)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer 訓練腳本')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                       help='選擇模式: train (訓練) 或 test (測試)')
    parser.add_argument('--push_to_hf', action='store_true',
                       help='訓練完成後推送到 Hugging Face Hub')
    parser.add_argument('--hf_repo_name', default='shakespeare-transformer',
                       help='Hugging Face 倉庫名稱')
    parser.add_argument('--hf_username', default='ewdlop', help='Hugging Face 用戶名')
    parser.add_argument('--hf_token', help='Hugging Face token (不建議在命令行中使用)')
    parser.add_argument('--hf_private', action='store_true',
                       help='創建私有 Hugging Face 倉庫')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("開始訓練 Transformer 模型...")
        model, dataset = train_model()
        print("訓練完成！")
        
        print("\n測試訓練好的模型:")
        generate_samples(model, dataset, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), num_samples=3)
        
        # 推送到 Hugging Face Hub
        if args.push_to_hf:
            print("\n" + "="*50)
            print("🤗 準備推送模型到 Hugging Face Hub...")
            print("="*50)
            
            try:
                from push_to_huggingface import push_to_huggingface
                
                # 安全地獲取 token
                username = args.hf_username
                token = args.hf_token
                
                if not token:
                    print("需要 Hugging Face token 來推送模型")
                    print("💡 提示: 您可以在 https://huggingface.co/settings/tokens 獲取 token")
                    token = getpass.getpass("請輸入您的 Hugging Face token (輸入時不會顯示): ")
                    
                    while not token.strip():
                        print("❌ Token 不能為空，請重新輸入:")
                        token = getpass.getpass("Token: ")
                
                push_to_huggingface(
                    model_path="transformer_model.pth",
                    repo_name=args.hf_repo_name,
                    username=username,
                    token=token,
                    private=args.hf_private,
                    commit_message=f"Add {args.hf_repo_name} - Shakespeare Transformer model trained on classical text"
                )
                
                print("🎉 模型已成功推送到 Hugging Face Hub!")
                
            except ImportError:
                print("❌ 無法導入 push_to_huggingface 模組")
                print("請確保 push_to_huggingface.py 文件存在")
            except Exception as e:
                print(f"❌ 推送到 Hugging Face 失敗: {str(e)}")
                print("您可以稍後手動運行:")
                print(f"python push_to_huggingface.py --repo_name {args.hf_repo_name} --interactive")
        
        elif input("\n是否要推送模型到 Hugging Face Hub? (y/N): ").lower().strip() == 'y':
            print("\n要推送到 Hugging Face，請運行:")
            print(f"python push_to_huggingface.py --repo_name {args.hf_repo_name} --interactive")
            print("或者重新運行訓練並添加 --push_to_hf 參數")
        
    elif args.mode == 'test':
        print("載入並測試模型...")
        test_model() 