from torch import nn
import torch



class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=0, output_size=6, max_length=20, **kws) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, 1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, 1, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim*2, hidden_dim, 1, bidirectional=True, batch_first=True)
        self.lin = nn.Linear(hidden_dim*2, output_size)
        self.out = nn.Softmax(1)

    def __str__(self) -> str:
        return 'LSTM'

    def forward(self, x):
        emb = self.emb(x)
        y, h = self.lstm1(emb)
        y, h = self.lstm2(y)
        y, h = self.lstm3(y)
        y = y[:, -1]
        y = self.lin(y)
        return self.out(y)
    
class Lin(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length=20, **kws):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(max_length*embedding_dim, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linearf = nn.Linear(128, 32)
        self.linearout = nn.Linear(32, 6)
        self.out = nn.Softmax(1)

    def __str__(self) -> str:
        return 'Linear'

    def forward(self, inputs):
        emb = self.embedding(inputs).view((inputs.shape[0], -1))
        x = nn.functional.relu(self.linear1(emb))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linearf(x))
        x = nn.functional.relu(self.linearout(x))
        return self.out(x)

class Conv(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length=20, **kws) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.c1 = nn.Conv1d(embedding_dim, 128, 3)
        self.c2 = nn.Conv1d(128, 128, 3)
        # self.mp = nn.MaxPool1d(3)
        self.b1 = nn.BatchNorm1d(128)
        self.c3 = nn.Conv1d(128, 64, 3)
        self.c4 = nn.Conv1d(64, 64, 3)
        self.dr = nn.Dropout(.2)
        self.b2 = nn.BatchNorm1d(64)
        self.c5 = nn.Conv1d(64, 64, 3)
        self.c6 = nn.Conv1d(64, embedding_dim, 3)
        self.intlin = nn.Linear(embedding_dim, 128)
        self.lin = nn.Linear(128, 6)
        self.out = nn.Softmax(1)

    def __str__(self) -> str:
        return 'Conv'

    def forward(self, x):
        emb = self.emb(x)
        emb = emb.permute(0, 2, 1) # batch, embedding_dim, seq length
        y = nn.functional.leaky_relu(self.c1(emb))
        y = nn.functional.leaky_relu(self.c2(y))
        # y = self.mp(y)
        y = self.b1(y)
        y = nn.functional.leaky_relu(self.c3(y))
        y = nn.functional.leaky_relu(self.c4(y))
        y = self.dr(y)
        y = self.b2(y)
        y = nn.functional.leaky_relu(self.c5(y))
        y = nn.functional.leaky_relu(self.c6(y))
        y, _ = y.max(dim=-1)
        # y = torch.flatten(y, start_dim=1)
        y = self.intlin(y)
        return self.out(self.lin(y))
    

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads=4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=heads, batch_first=True)
        self.n1 = nn.LayerNorm(emb_dim)
        self.n2 = nn.LayerNorm(emb_dim)
        self.dp = nn.Dropout(.2)

        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            nn.ReLU(),
            nn.Linear(4*emb_dim, emb_dim)
        )

    def forward(self, x):
        att, _ = self.attention(x, x, x)
        att = self.dp(att)
        x = self.n1(att + x)
        ff = self.ff(x)
        ff = self.dp(ff)
        return self.n2(ff + x)
    
class Transformer(nn.Module):
    def __init__(self, embedding_dim, vocab_size, depth=4, heads=4, max_length=20, **kws) -> None:
        super().__init__()
        self.max_length = max_length
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(max_length, embedding_dim)
        trans_blocks = []
        for d in range(depth):
            trans_blocks.append(TransformerBlock(embedding_dim, heads))
        self.trans_blocks = nn.Sequential(*trans_blocks)
        self.lin = nn.Linear(embedding_dim, 6)
        self.out = nn.Softmax(1)

    def __str__(self) -> str:
        return 'Transformer'

    def forward(self, x):
        tokens = self.token_emb(x)
        b, s, e = tokens.shape
        positions = torch.arange(s).to('cuda') # batch, seq_length, emb dim
        positions = self.pos_emb(positions).expand(b, s, e)
        # [None, :, :].expand(b, s, e)
        # print(tokens.shape, positions.shape)
        x = tokens + positions
        x = self.trans_blocks(x)

        x = self.lin(x.mean(dim=1))
        return self.out(x)
    
