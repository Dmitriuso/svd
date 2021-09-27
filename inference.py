import numpy as np
import random
import spacy
import torch

import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field

from pathlib import Path

from utils.pre_processing import SRC, TRG

work_dir = Path(__file__).parent.resolve()
models_dir = work_dir / "models"

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class TranslationInference:
    def __init__(self, model_path, src_field, trg_field, max_len, device):
        self.model = torch.load(model_path, map_location=device)
        self.src_field = src_field
        self.trg_field = trg_field
        self.max_len = max_len
        self.device = device

    def inference(self, sentence):
        model.eval()

        if isinstance(sentence, str):
            nlp = spacy.load('de_core_news_sm')
            tokens = [token.text.lower() for token in nlp(sentence)]
        else:
            tokens = [token.lower() for token in sentence]

        tokens = [src_field.init_token] + tokens + [src_field.eos_token]

        src_indexes = [src_field.vocab.stoi[token] for token in tokens]

        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

        src_mask = model.make_src_mask(src_tensor)

        with torch.no_grad():
            enc_src = model.encoder(src_tensor, src_mask)

        trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

        for i in range(max_len):

            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

            trg_mask = model.make_trg_mask(trg_tensor)

            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            pred_token = output.argmax(2)[:,-1].item()

            trg_indexes.append(pred_token)

            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break

        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

        return trg_tokens[1:], attention


def translate():
    sentence = "I habe aber alles verstanden."
    max_len = 50
    model_path = models_dir / "translate_svd_de_en.pt"
    src_field = SRC
    trg_field = TRG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    translation = TranslationInference(
        model_path=model_path,
        src_field=src_field,
        trg_field=trg_field,
        max_len=max_len,
        device=device
    )
    print(f'prediction : {translation.inference(sentence)}')
    pass


if __name__ == "__main__":
    translate()
