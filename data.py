import torch
from torch.utils.data import Dataset




class SentDataset(Dataset):
    def __init__(self, df, vocab, max_size) -> None:
        self.df = df
        self.map = vocab
        self.max_size = max_size

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, ind):
        # 35 is max
        text, _, label = self.df.iloc[ind]
        # manually pad
        ptext = [self.map[i] for i in text.split(' ')]
        seq = ptext.copy()
        if len(ptext) <= 10:
            times = 20 // len(ptext)
            for i in range(times):
                seq += ptext

        if len(seq) <= self.max_size:
            seq.extend([0] * (self.max_size - len(seq)))
        else:
            seq = seq[:self.max_size]
        text_tensor = torch.LongTensor(seq)
        label_tensor = torch.LongTensor([label])
        return text_tensor, label_tensor
    