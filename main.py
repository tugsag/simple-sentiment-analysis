import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from utils import get_and_preprocess
from data import SentDataset
from model import LSTM, Conv, Lin, Transformer




def plot_loss(train_loss, val_loss, trac, valac):
    # Looks ugly but whatever
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    epochs = range(len(train_loss))
    ax[0].plot(epochs, train_loss, color='b', label='train')
    ax[0].plot(epochs, val_loss, color='r', label='val')
    ax[0].set_ylabel('loss')
    ax[0].set_title('loss')
    ax[0].legend()

    ax[1].plot(epochs, trac, color='b', label='train')
    ax[1].plot(epochs, valac, color='r', label='val')
    ax[1].set_ylabel('acc')
    ax[1].set_title('acc')
    ax[1].legend()
    return f

def save_model(model, path, optimize=False):
    torch.save(model.state_dict(), path)
    if optimize:
        example = torch.randint(0, 1000, (1, 30))
        model.eval()
        model = model.to('cpu')
        # cmod = torch.quantization.convert(model)
        # scripted_model = torch.jit.script(cmod)
        scripted_model = torch.jit.trace(model, example)
        opt_model = optimize_for_mobile(scripted_model)
        opt_model._save_for_lite_interpreter('models/mobile_model.ptl')

def train(epochs, arch, dev='cuda', max_length=35):
    data, vocab = get_and_preprocess()
    print(f'vocab size is: {len(vocab)}')

    train, test = train_test_split(data, test_size=0.2, stratify=data['label_enc'])
    train_loader = DataLoader(SentDataset(train, vocab, max_length), batch_size=64, num_workers=8)
    test_loader = DataLoader(SentDataset(test, vocab, max_length), batch_size=64, num_workers=8)

    model = arch(vocab_size=len(vocab), 
                 embedding_dim=32, 
                 hidden_dim=48,
                 max_length=max_length).to(dev)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    tl, vl = [], []
    trac, valac = [], []
    for e in tqdm(range(epochs)):
        model.train()
        training_loss = 0
        trouts, trlabels = [], []
        for tr in train_loader:
            text, label = tr
            optimizer.zero_grad()
            outs = model(text.to(dev))
            trouts.append(outs.detach().cpu())
            trlabels.append(label)
            loss = criterion(outs, label.squeeze(1).to(dev))
            loss.backward()
            optimizer.step()
            training_loss += loss
        # print(f"Training loss: {training_loss/len(train_loader)}")
        tot_out = torch.cat(trouts)
        tot_label = torch.cat(trlabels)
        ac = accuracy_score(tot_label.squeeze(1).numpy(), torch.argmax(tot_out, dim=1).numpy())
        trac.append(ac)
        tl.append((training_loss/len(train_loader)).detach().cpu())
        # validation
        model.eval()
        validation_loss = 0
        valouts, vallabels = [], []
        for va in test_loader:
            text, label = va
            outs = model(text.to(dev))
            valouts.append(outs.detach().cpu())
            vallabels.append(label)
            loss = criterion(outs, label.squeeze(1).to(dev))
            validation_loss += loss
        # print(f"Validation loss: {validation_loss/len(test_loader)}")
        if e == 0:
            print(torch.argmax(outs[:10], dim=1), label[:10])
        tot_out = torch.cat(valouts)
        tot_label = torch.cat(vallabels)
        ac = accuracy_score(tot_label.squeeze(1).numpy(), torch.argmax(tot_out, dim=1).numpy())
        valac.append(ac)
        vl.append((validation_loss/len(test_loader)).detach().cpu())

    ## Save last model
    save_model(model, f'models/last_model_{str(model)}.ptl', optimize=True)
    f = plot_loss(tl, vl, trac, valac)
    f.savefig(f'figs/train_info_{max_length}_{str(model)}.png')
    print(torch.argmax(outs[:10], dim=1), label[:10])

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', default=10, type=int)
    parser.add_argument('-l', default=35, type=int)
    parser.add_argument('-m', default=0, type=int)
    parser.add_argument('-a', default=1, type=int)
    args = parser.parse_args()
    archs = {
        3: Transformer,
        2: Lin,
        1: Conv,
        0: LSTM
    }

    if args.a:
        for i in range(len(archs)):
            train(args.e, archs[i], max_length=args.l)
    else:
        train(args.e, archs[args.m], max_length=args.l)