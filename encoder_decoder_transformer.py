import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Using PyTorch's built-in MultiheadAttention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=False  # PyTorch default: (seq_len, batch, embed_dim)
        )
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None):
        # x shape: (seq_len, batch_size, d_model)
        
        # Multi-head self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(
            query=x, 
            key=x, 
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Masked self-attention
        self.masked_self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=False
        )
        # Cross-attention (decoder attends to encoder)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=False
        )
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        # x shape: (tgt_seq_len, batch_size, d_model)
        # enc_output shape: (src_seq_len, batch_size, d_model)
        
        # Masked multi-head self-attention
        attn_output, _ = self.masked_self_attention(
            query=x,
            key=x,
            value=x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Multi-head cross-attention (decoder attends to encoder)
        attn_output, _ = self.cross_attention(
            query=x,
            key=enc_output,
            value=enc_output,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_encoder_layers=6, n_decoder_layers=6, d_ff=2048, dropout=0.1, pad_idx=0):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encodings
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq):
        """Create padding mask for sequences (True for padding tokens)"""
        return seq == self.pad_idx
    
    def create_look_ahead_mask(self, size):
        """Create look-ahead mask for decoder (upper triangular matrix)"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()
    
    def encode(self, src, src_key_padding_mask=None):
        """Encode source sequence"""
        # src shape: (batch_size, src_seq_len)
        # Convert to (src_seq_len, batch_size, d_model)
        
        # Source embedding + positional encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)  # (batch, seq, d_model)
        src_emb = src_emb.transpose(0, 1)  # (seq, batch, d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        
        # Pass through encoder layers
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, key_padding_mask=src_key_padding_mask)
            
        return enc_output
    
    def decode(self, tgt, enc_output, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        """Decode target sequence"""
        # tgt shape: (batch_size, tgt_seq_len)
        # Convert to (tgt_seq_len, batch_size, d_model)
        
        # Target embedding + positional encoding
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)  # (batch, seq, d_model)
        tgt_emb = tgt_emb.transpose(0, 1)  # (seq, batch, d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # Pass through decoder layers
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(
                dec_output, 
                enc_output, 
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
        return dec_output
    
    def forward(self, src, tgt):
        """Forward pass"""
        # src shape: (batch_size, src_seq_len)
        # tgt shape: (batch_size, tgt_seq_len)
        
        batch_size, src_seq_len = src.shape
        batch_size, tgt_seq_len = tgt.shape
        
        # Create masks
        src_key_padding_mask = self.create_padding_mask(src)  # (batch, src_seq)
        tgt_key_padding_mask = self.create_padding_mask(tgt)  # (batch, tgt_seq)
        tgt_mask = self.create_look_ahead_mask(tgt_seq_len).to(tgt.device)  # (tgt_seq, tgt_seq)
        
        # Encode
        enc_output = self.encode(src, src_key_padding_mask)
        
        # Decode
        dec_output = self.decode(
            tgt, 
            enc_output, 
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Final linear transformation
        # Convert back to (batch, seq, d_model)
        dec_output = dec_output.transpose(0, 1)
        output = self.linear(dec_output)
        
        # Apply softmax to get probabilities
        output_probs = F.softmax(output, dim=-1)
        
        return output_probs
    
    def generate(self, src, max_len=50, start_token=1, end_token=2):
        """Generate sequence using greedy decoding"""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Encode source
        src_key_padding_mask = self.create_padding_mask(src)
        enc_output = self.encode(src, src_key_padding_mask)
        
        # Initialize target with start token
        tgt = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
        
        for i in range(max_len - 1):
            # Create masks for current target
            tgt_key_padding_mask = self.create_padding_mask(tgt)
            tgt_mask = self.create_look_ahead_mask(tgt.size(1)).to(device)
            
            # Decode
            dec_output = self.decode(
                tgt, 
                enc_output, 
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            # Get next token probabilities
            dec_output = dec_output.transpose(0, 1)  # (batch, seq, d_model)
            next_token_logits = self.linear(dec_output[:, -1, :])  # (batch, vocab_size)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch, 1)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Check if all sequences have generated end token
            if (next_token == end_token).all():
                break
                
        return tgt