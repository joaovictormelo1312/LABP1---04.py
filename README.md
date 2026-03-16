# Laboratório 4 — Implementação de Transformer "From Scratch"

## Descrição

Este projeto apresenta a implementação de um **modelo Transformer Encoder–Decoder completo utilizando PyTorch**, desenvolvido a partir dos conceitos fundamentais da arquitetura proposta no artigo *"Attention is All You Need"* (Vaswani et al., 2017).

Diferente de implementações prontas disponíveis em bibliotecas como `torch.nn.Transformer`, este projeto constrói manualmente os principais componentes da arquitetura, permitindo compreender detalhadamente o funcionamento interno do modelo.

O sistema foi testado utilizando uma frase simples de entrada ("Thinking Machines") para demonstrar o fluxo completo de processamento entre encoder e decoder.

---

# Objetivo

O objetivo deste laboratório é implementar e compreender os seguintes elementos fundamentais da arquitetura Transformer:

- Mecanismo de **Scaled Dot-Product Attention**
- **Positional Encoding**
- Camadas **Feed Forward Network**
- Estrutura **Add & Norm**
- **Encoder Block**
- **Decoder Block**
- Processo de **geração auto-regressiva**

---

# Estrutura do Projeto
.
│
├── transformer_from_scratch.py # Implementação completa do Transformer
├── README.md # Documentação do projeto

---

# Arquitetura do Transformer

A arquitetura Transformer é composta por duas partes principais:

### Encoder

O encoder recebe a sequência de entrada e gera representações contextuais para cada token.

Cada **Encoder Block** possui:

1. **Self-Attention**
2. **Add & Norm**
3. **Feed Forward Network**
4. **Add & Norm**

A saída final do encoder é uma representação contextual chamada **Z**, que será utilizada pelo decoder.

---

### Decoder

O decoder recebe os tokens já gerados e utiliza as representações do encoder para prever o próximo token.

Cada **Decoder Block** possui:

1. **Masked Self-Attention**
2. **Add & Norm**
3. **Cross Attention** (atenção sobre a saída do encoder)
4. **Add & Norm**
5. **Feed Forward Network**
6. **Add & Norm**

---

# Scaled Dot-Product Attention

O mecanismo central do Transformer é a atenção, calculada pela fórmula:

Attention(Q,K,V) = softmax(QKᵀ / √d_k) V

Onde:

- **Q (Query)** representa o token atual
- **K (Key)** representa os tokens de referência
- **V (Value)** contém as representações utilizadas para compor a saída

O termo √d_k é utilizado para estabilizar os valores da softmax.

---

# Positional Encoding

Como o Transformer não possui recorrência ou convolução, é necessário adicionar informação sobre a posição dos tokens na sequência.

Isso é feito através do **Positional Encoding**, que utiliza funções senoidais:

PE(pos,2i) = sin(pos / 10000^(2i/d_model))
PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))

Essa estratégia permite ao modelo capturar relações entre posições relativas na sequência.

---

# Feed Forward Network

Cada posição do Transformer passa por uma rede neural totalmente conectada:

FFN(x) = max(0, xW1 + b1)W2 + b2


Essa rede expande a dimensionalidade da representação e depois retorna para a dimensão original.

---

# Add & Norm

Cada subcamada utiliza uma conexão residual seguida de normalização:

LayerNorm(x + Sublayer(x))


Esse mecanismo melhora a estabilidade do treinamento e facilita a propagação de gradientes.

---

# Processo de Geração Auto-Regressiva

O decoder gera tokens sequencialmente utilizando um processo auto-regressivo:

1. O processo inicia com o token `<START>`
2. O modelo prevê o próximo token da sequência
3. O token gerado é inserido novamente como entrada do decoder
4. O processo continua até que o token `<EOS>` seja gerado ou o limite máximo seja atingido

Fluxo de geração:

<START> → token1 → token2 → token3 → ... → <EOS>

---

# Exemplo de Execução

Entrada fornecida ao encoder:

Thinking Machines

Fluxo de geração do decoder:

<START> → ... → <EOS>


Como o modelo não foi treinado, os tokens gerados servem apenas para demonstrar o funcionamento da arquitetura.

---

# Tecnologias Utilizadas

- Python
- PyTorch
- Torch.nn
- Torch.nn.functional

---

# Referência

Vaswani, A. et al. (2017).  
**Attention is All You Need**.  
Advances in Neural Information Processing Systems (NeurIPS).

---

# Observação sobre uso de IA

Ferramentas de inteligência artificial foram utilizadas como apoio no desenvolvimento da implementação.  
Todo o código foi revisado, compreendido e validado manualmente antes da submissão.

---

# Autor

Aluno: **[SEU NOME]**  
Disciplina: **Processamento de Linguagem Natural**




