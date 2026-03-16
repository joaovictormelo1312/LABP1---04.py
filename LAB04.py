import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. FUNÇÕES BÁSICAS

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_q, d_model)
    K: (batch, seq_k, d_model)
    V: (batch, seq_k, d_model)
    mask: (seq_q, seq_k) ou (batch, seq_q, seq_k)
          posições mascaradas devem estar como 0
    """
    d_k = Q.size(-1)

    # scores = QK^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # onde mask == 0, coloca -inf
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (1, seq_q, seq_k)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


def positional_encoding(seq_len, d_model, device):
    """
    Gera Positional Encoding senoidal.
    Retorna tensor de shape (1, seq_len, d_model)
    """
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)


def generate_causal_mask(seq_len, device):
    """
    Máscara causal:
    permite olhar apenas para a posição atual e anteriores.
    shape: (seq_len, seq_len)
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()


# 2. BLOCOS FUNDAMENTAIS

class PositionWiseFFN(nn.Module):
    """
    FFN com expansão de dimensão e ReLU:
    d_model -> d_ff -> d_model
    """
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class AddNorm(nn.Module):
    """
    Output = LayerNorm(x + Sublayer(x))
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)


class SelfAttentionBlock(nn.Module):
    """
    Projeta Q, K, V a partir de x e aplica atenção.
    """
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        output, attn = scaled_dot_product_attention(Q, K, V, mask)
        return output, attn


class CrossAttentionBlock(nn.Module):
    """
    Q vem de y, K e V vêm de Z (saída do encoder).
    """
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, y, Z, mask=None):
        Q = self.Wq(y)
        K = self.Wk(Z)
        V = self.Wv(Z)

        output, attn = scaled_dot_product_attention(Q, K, V, mask)
        return output, attn


# 3. ENCODER BLOCK

class EncoderBlock(nn.Module):
    """
    Fluxo:
    1. Self-Attention
    2. Add & Norm
    3. FFN
    4. Add & Norm
    """
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model)
        self.add_norm1 = AddNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x)
        x = self.add_norm1(x, attn_output)

        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)

        return x


class Encoder(nn.Module):
    """
    Pilha de múltiplos EncoderBlocks
    """
    def __init__(self, vocab_size, d_model=512, d_ff=2048, num_layers=2, max_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, d_ff) for _ in range(num_layers)
        ])
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + positional_encoding(seq_len, self.d_model, device)

        for layer in self.layers:
            x = layer(x)

        return x  # Z

# 4. DECODER BLOCK

class DecoderBlock(nn.Module):
    """
    Fluxo:
    1. Masked Self-Attention
    2. Add & Norm
    3. Cross-Attention
    4. Add & Norm
    5. FFN
    6. Add & Norm
    """
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.masked_self_attn = SelfAttentionBlock(d_model)
        self.add_norm1 = AddNorm(d_model)

        self.cross_attn = CrossAttentionBlock(d_model)
        self.add_norm2 = AddNorm(d_model)

        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.add_norm3 = AddNorm(d_model)

    def forward(self, y, Z, causal_mask=None):
        masked_attn_output, _ = self.masked_self_attn(y, causal_mask)
        y = self.add_norm1(y, masked_attn_output)

        cross_attn_output, _ = self.cross_attn(y, Z)
        y = self.add_norm2(y, cross_attn_output)

        ffn_output = self.ffn(y)
        y = self.add_norm3(y, ffn_output)

        return y


class Decoder(nn.Module):
    """
    Pilha de múltiplos DecoderBlocks + projeção final para vocabulário
    """
    def __init__(self, vocab_size, d_model=512, d_ff=2048, num_layers=2, max_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, d_ff) for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, target_ids, Z):
        # target_ids: (batch, tgt_seq_len)
        batch_size, tgt_seq_len = target_ids.shape
        device = target_ids.device

        y = self.embedding(target_ids) * math.sqrt(self.d_model)
        y = y + positional_encoding(tgt_seq_len, self.d_model, device)

        causal_mask = generate_causal_mask(tgt_seq_len, device)

        for layer in self.layers:
            y = layer(y, Z, causal_mask)

        logits = self.output_linear(y)              # (batch, tgt_seq_len, vocab_size)
        probs = F.softmax(logits, dim=-1)

        return logits, probs

# 5. TRANSFORMER COMPLETO

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, d_ff=2048, num_layers=2, max_len=50):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, d_ff, num_layers, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, d_ff, num_layers, max_len)

    def forward(self, encoder_input_ids, decoder_input_ids):
        Z = self.encoder(encoder_input_ids)
        logits, probs = self.decoder(decoder_input_ids, Z)
        return logits, probs

# 6. EXEMPLO DE USO COM FRASE "Thinking Machines"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vocabulário toy
    vocab = {
        "<PAD>": 0,
        "<START>": 1,
        "<EOS>": 2,
        "Thinking": 3,
        "Machines": 4,
        "Máquinas": 5,
        "Pensantes": 6,
        "Inteligentes": 7
    }

    id_to_token = {idx: tok for tok, idx in vocab.items()}

    src_vocab_size = len(vocab)
    tgt_vocab_size = len(vocab)

    # Instancia o modelo
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=64,      # menor para facilitar teste
        d_ff=128,
        num_layers=2,
        max_len=20
    ).to(device)

    # Frase de entrada simulando "Thinking Machines"
    encoder_input = torch.tensor([[vocab["Thinking"], vocab["Machines"]]], device=device)

    # 7. LOOP AUTO-REGRESSIVO
  
    model.eval()

    generated = [vocab["<START>"]]
    max_steps = 10

    with torch.no_grad():
        for _ in range(max_steps):
            decoder_input = torch.tensor([generated], device=device)

            logits, probs = model(encoder_input, decoder_input)

            # pega a distribuição da última posição
            next_token_probs = probs[0, -1, :]
            next_token_id = torch.argmax(next_token_probs).item()

            generated.append(next_token_id)

            if next_token_id == vocab["<EOS>"]:
                break

    generated_tokens = [id_to_token[idx] for idx in generated]

    print("Entrada do Encoder:")
    print([id_to_token[idx.item()] for idx in encoder_input[0]])

    print("\nSaída gerada pelo Decoder:")
    print(generated_tokens)


if __name__ == "__main__":
    main()
