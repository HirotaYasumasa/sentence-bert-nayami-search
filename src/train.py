import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sentence_transformers import losses, SentenceTransformer, models, InputExample

import random
import unicodedata
import json
from datetime import datetime

from pathlib import Path
from typing import List
from classopt import classopt

@classopt(default_long=True)
class Args:
  model_name: str = 'cl-tohoku/bert-base-japanese-v3'

  dataset_dir: Path = './datasets/nayami'
  input_dir: Path = './data/nayami'

  max_seq_len: int = 256
  epochs: int = 1
  pooling: str = 'mean'
  device: str = 'cuda:0'
  seed: int = 42

  #細分類=1, 小分類=2, 中分類=3, 大分類=4
  sizuoka_class: int = 1
  test_size: float = 0.30

  def __post_init__(self):
    dataset_dir = Path(self.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    model_name = self.model_name.replace("/", "__")
    self.model_dir = Path("models") / model_name / date
    self.model_dir.mkdir(parents=True, exist_ok=True)

def make_dataloader(df: pd.DataFrame) -> DataLoader:
  label_to_texts: dict[int, List[str]] = {}
  train_examples = []

  for _, row in tqdm(df.iterrows(), desc='Separate label and content', total=len(df)):
    label:int = row['labels']
    text:str = row['content']
    if label not in label_to_texts:
      label_to_texts[label] = []
    label_to_texts[label].append(text)

  for label, texts in tqdm(label_to_texts.items(), desc='Creating Text Pairs'):
    # 同じラベルのテキストペアを追加
    for i in range(len(texts)):
      for j in range(i + 1, len(texts)):
        train_examples.append(InputExample(texts=[texts[i], texts[j]], label=1.0))

    # 異なるラベルのテキストペアを追加
    other_labels = list(label_to_texts.keys())
    other_labels.remove(label)
    for other_label in other_labels:
      for other_text in label_to_texts[other_label]:
        train_examples.append(InputExample(texts=[random.choice(texts), other_text], label=0.0))

  train_dataloader: DataLoader = DataLoader(train_examples, shuffle=True, batch_size=16)

  return train_dataloader

def split_category_id(category_id: str, class_num: int):
  category = category_id.split('.')
  for _ in range(class_num):
    if category:
      category.pop()
  return '.'.join(category)

def preprocess_df(nayami_df: pd.DataFrame) -> DataLoader:
  nayami_df['content'] = nayami_df['content'].map(lambda x: unicodedata.normalize('NFKC', x))
  nayami_df['content'] = nayami_df['content'].map(lambda x: x.replace('\n', ''))

  nayami_df['category_id'] = nayami_df['category_id'].map(lambda x: split_category_id(x, args.sizuoka_class))
  nayami_df['labels'] = nayami_df['category_id'].astype('category').cat.codes

  train_df, test_df = train_test_split(nayami_df, test_size=args.test_size, stratify=nayami_df['labels'], random_state=args.seed)

  train_df.to_json(args.dataset_dir / 'train.jsonl', orient='records', lines=True, force_ascii=False)
  test_df.to_json(args.dataset_dir / 'test.jsonl', orient='records', lines=True, force_ascii=False)

def args_to_dict(args):
  args_dict = vars(args).copy()
  for key, value in args_dict.items():
    if isinstance(value, Path):
      args_dict[key] = str(value)
  return args_dict

def main(args: Args):
  random.seed(args.seed)

  nayami_df: pd.DataFrame[{'id': int, 'type': str, 'category_id': str, 'content': str}] = pd.read_csv(args.input_dir / 'sizuoka_nayami_subclass.csv')

  preprocess_df(nayami_df)

  train_df: pd.DataFrame[{'id': int, 'type': str, 'category_id': str, 'content': str, 'labels': int}] = pd.read_json(args.dataset_dir / 'train.jsonl', orient='records', lines=True)

  embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_len)
  pooling_model = models.Pooling(embedding_model.get_word_embedding_dimension(), pooling_mode=args.pooling)

  model = SentenceTransformer(modules=[embedding_model, pooling_model]).to(args.device)
  loss = losses.CosineSimilarityLoss(model)

  train_dataloader = make_dataloader(train_df)

  with open(args.model_dir / 'config.json', 'w', encoding='utf-8') as f:
    json.dump(args_to_dict(args), f, ensure_ascii=False, indent=2)

  model.fit(train_objectives=[(train_dataloader, loss)], epochs=args.epochs, warmup_steps=100, optimizer_class=torch.optim.AdamW, optimizer_params={'lr': 2e-5}, weight_decay=0.01, evaluation_steps=100, output_path=args.model_dir)

if __name__ == '__main__':
  args = Args.from_args()
  main(args)
