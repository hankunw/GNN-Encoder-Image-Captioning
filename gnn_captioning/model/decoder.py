
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class ImageCaptioningModel(nn.Module):
    def __init__(self,
                 encoder,
                 vocab_size,
                 max_seq_len,
                 encoder_dim = 128,
                 embed_dim=128,
                 num_heads=8,
                 num_layers=6,
                 dropout=0.1,
                 encoding_projection = True,
                 sin_pe = True
                 ):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.sin_pe = sin_pe
        self.encoding_projection = encoding_projection

        if encoding_projection:
            self.encoder_proj = nn.Linear(encoder_dim, embed_dim)
        else:
            assert encoder_dim == embed_dim, "encoder output dim different from embed dim"

        # Text embedding layer (+ position encoding)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        if sin_pe:
            self.register_buffer(
                "position_encoding",
                self.create_positional_encoding(max_seq_len, embed_dim),
                persistent=False  # 不保存到state_dict
            )
        else:
            self.position_encoding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=dropout,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final projection
        self.output_layer = nn.Linear(embed_dim, vocab_size)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # 初始化嵌入层
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)

        if not self.sin_pe:  # 仅在可学习位置编码时初始化
            nn.init.normal_(self.position_encoding.weight, mean=0, std=0.02)

        # 初始化输出层
        nn.init.xavier_normal_(self.output_layer.weight)
        self.output_layer.bias.data.zero_()

    @staticmethod
    def create_positional_encoding(max_len, d_model):
        """创建正弦位置编码矩阵"""
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model//2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列用cos
        return pe  # (max_len, d_model)
    def forward(self, img, caption_tokens, pad_mask=None):
        # 编码图像
        memory = self.encoder(img)  # [batch, num_patches, encoder_dim]

        # 投影层（如果启用）
        if self.encoding_projection:
            memory = self.encoder_proj(memory)  # [batch, num_patches, embed_dim]

        memory = memory.permute(1, 0, 2)  # [num_patches, batch, embed_dim]

        # 处理文本输入
        batch_size, seq_len = caption_tokens.shape
        device = caption_tokens.device

        # 位置编码
        if self.sin_pe:
            pos_embeds = self.position_encoding[:seq_len].unsqueeze(0)  # [1, seq_len, embed_dim]
        else:
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
            pos_embeds = self.position_encoding(pos_ids)  # [1, seq_len, embed_dim]

        # 文本嵌入（统一缩放）
        token_embeds = self.token_embedding(caption_tokens)  # [batch, seq_len, embed_dim]
        tgt = token_embeds * math.sqrt(self.embed_dim) + pos_embeds
        tgt = tgt.permute(1, 0, 2)  # [seq_len, batch, embed_dim]

        # 生成mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)

        # Transformer解码
        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=pad_mask
        )  # (seq_len, batch, embed_dim)

        # 投影到词汇表
        decoded = decoded.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
        logits = self.output_layer(decoded)

        return logits

    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码（防止看到未来信息）"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """来自 HuggingFace 的采样过滤函数"""
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        return logits
    def generate(self, img, max_length, tokenizer, temperature=1.0, top_k=50, top_p=0.95, decoding_strategy = 'greedy'):
        """适配 GPT-2 Tokenizer 的生成函数"""
        self.eval()
        device = img.device
        batch_size = img.shape[0]

        with torch.no_grad():
            # ============== image encoding ==============
            memory = self.encoder(img)  # [batch, num_patches, encoder_dim]

            # projection
            if self.encoding_projection:
                memory = self.encoder_proj(memory)  # [batch, num_patches, embed_dim]

            memory = memory.permute(1, 0, 2)  # [num_patches, batch, embed_dim]

            # initialize
            output = torch.full((batch_size, 1),
                            tokenizer.eos_token_id,
                            dtype=torch.long,
                            device=device)

            for _ in range(max_length):
                seq_len = output.size(1)

                token_embeds = self.token_embedding(output)  # [batch, seq_len, embed_dim]

                # add positional encoding:
                if self.sin_pe:
                    pos_embeds = self.position_encoding[:seq_len].unsqueeze(0)
                else:
                    pos_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
                    pos_embeds = self.position_encoding(pos_ids)  # [1, seq_len, embed_dim]

                # ===== Transformer 解码 =====
                # 生成因果掩码（防止看到未来信息）
                tgt = token_embeds* math.sqrt(self.embed_dim)+pos_embeds
                tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)

                # 解码器前向传播
                decoded = self.decoder(
                    tgt.permute(1, 0, 2),  # [seq_len, batch, embed_dim]
                    memory,
                    tgt_mask=tgt_mask
                ).permute(1, 0, 2)  # 恢复为 [batch, seq_len, embed_dim]

                logits = self.output_layer(decoded[:, -1, :])  # [batch, vocab_size]

                # 应用温度采样
                logits = logits / temperature
                filtered_logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)

                # 采样下一个 Token
                if decoding_strategy == 'greedy':
                    next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
                elif decoding_strategy == 'multinomial':
                    next_token = torch.multinomial(probabilities, num_samples=1)  # [batch, 1]
                # ===== 终止条件判断 =====
                output = torch.cat([output, next_token], dim=1)

                # 所有样本都生成 EOS 则提前终止
                if (next_token == tokenizer.eos_token_id).all():
                    break

        # ============== 后处理 ==============
        # 解码时跳过起始的 EOS（根据需求调整）
        return output[:, 1:] if output.size(1) > 1 else output