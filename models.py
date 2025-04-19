import os
import pickle
import logging
import torch
import pandas as pd
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmailDataset(Dataset):
    """Dataset for email classification."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class EmailClassifier:
    """BERT-based email classifier."""

    def __init__(self, device=None, model_dir='./model_data'):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_mapping = {}
        self.inverse_label_mapping = {}
        self.model = None

        os.makedirs(model_dir, exist_ok=True)

    def prepare_data(self, data_csv: str) -> Tuple[List[str], List[int]]:
        logger.info(f"Loading data from {data_csv}")
        df = pd.read_csv(data_csv)

        if 'email' not in df.columns or 'type' not in df.columns:
            raise ValueError("CSV must have 'email' and 'type' columns")

        df = df.dropna(subset=['email', 'type'])

        unique_labels = df['type'].unique()
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}

        with open(os.path.join(self.model_dir, 'label_mapping.pkl'), 'wb') as f:
            pickle.dump({
                'label_mapping': self.label_mapping,
                'inverse_label_mapping': self.inverse_label_mapping
            }, f)

        texts = df['email'].tolist()
        labels = [self.label_mapping[label] for label in df['type']]

        return texts, labels

    def train(self, data_csv: str, epochs=4, batch_size=16, learning_rate=2e-5):
        texts, labels = self.prepare_data(data_csv)

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        train_dataset = EmailDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = EmailDataset(val_texts, val_labels, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        num_labels = len(self.label_mapping)
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=num_labels
        )
        self.model.to(self.device)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        best_val_accuracy = 0

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Average training loss: {avg_loss:.4f}")

            # Validation
            self.model.eval()
            val_preds, val_true = [], []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(labels.cpu().numpy())

            val_accuracy = accuracy_score(val_true, val_preds)
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                logger.info(f"Saving best model with accuracy: {val_accuracy:.4f}")
                self.save_model()

                report = classification_report(
                    val_true, val_preds,
                    target_names=[self.inverse_label_mapping[i] for i in range(num_labels)],
                    digits=4
                )
                logger.info(f"Classification Report:\n{report}")

        return best_val_accuracy

    def save_model(self):
        if self.model:
            self.model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            logger.info(f"Model saved to {self.model_dir}")
        else:
            logger.warning("No model to save.")

    def load_model(self):
        try:
            with open(os.path.join(self.model_dir, 'label_mapping.pkl'), 'rb') as f:
                mappings = pickle.load(f)
                self.label_mapping = mappings['label_mapping']
                self.inverse_label_mapping = mappings['inverse_label_mapping']

            self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded from {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict(self, text: str) -> str:
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not loaded. Please train or load a model first.")

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()

        return self.inverse_label_mapping[pred]
