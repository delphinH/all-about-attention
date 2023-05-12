import torch
import torch.nn as nn
import torch.optim as optim

import spacy

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from utils import translate_sentence, bleu

import sys

spacy_eng = spacy.load('en_core_web_sm')
spacy_ger = spacy.load('nl_core_news_sm')


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

english = Field(tokenize=tokenize_eng, lower=True, init_token='<sos>', eos_token='<eos>')
german = Field(tokenize=tokenize_ger, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))

# Examine the data in Multi30k one by one
# for i in range(10):
#     print(test_data[i].__dict__.values())

# build_vocab may numericalize the tokens
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Transformer(nn.Module):
    def __init__(
            self,
            embed_size,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_len,
            device
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.src_position_embedding = nn.Embedding(max_len, embed_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.trg_position_embedding = nn.Embedding(max_len, embed_size)
        self.device = device

        self.transformer = nn.Transformer(
            embed_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # src shape: (src_len, N)
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # (N, src_len)
        return src_mask

    def forward(self, src, trg):
        src_seq_len, N = src.size()
        trg_seq_len, N = trg.size()

        src_position = (torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N).to(self.device))
        trg_position = (torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N).to(self.device))

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_position))
        )

        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_position))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = src_padding_mask,
            tgt_mask = trg_mask
        )

        out = self.fc_out(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training hyperparameters
num_epochs = 9
lr = 2e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embed_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.1
max_len = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi['<pad>']

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

# checking the data
# for batch in test_iter:
#     print(batch)
#     print(batch.trg)


model = Transformer(
    embed_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

src_sentence = "Das Mädchen, das eine Orange aß, ritt auf einem Pferd." # The girl who ate an orange rode a horse.

for epoch in range(num_epochs):
    print(f'Epoch {epoch} / {num_epochs-1}')

    model.eval()
    translated_sen = translate_sentence(model, src_sentence, german, english, device, max_length=50)
    print(f'Ger to Eng \n {translated_sen}')

    model.train()
    losses = []
    for batch_idx, batch in enumerate(train_iter):
        input_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(input_data, target[:-1])
        output = output.reshape(-1, output.size(2))
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= 1)

        optimizer.step()
    mean_loss = sum(losses) / len(losses)

print(mean_loss)

sys.exit()
score = bleu(test_data, model, german, english, device)
print(f'bleu score {score*100:.2f}')
