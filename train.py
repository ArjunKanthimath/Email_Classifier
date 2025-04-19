import argparse
import logging
import torch
from models import EmailClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path, epochs=4, batch_size=16, learning_rate=2e-5):
    """
    Train the email classification model.

    Args:
        data_path: Path to the CSV file containing emails and their categories
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
    """
    logger.info(f"Training model with data from {data_path}")

    # Detect GPU/CPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"📦 Using device: {device}")

    # Initialize classifier with device
    classifier = EmailClassifier(device=device)

    # Train model
    accuracy = classifier.train(
        data_csv=data_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    logger.info(f"✅ Training completed. Best validation accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train email classification model")
    parser.add_argument("--data", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")

    args = parser.parse_args()

    train_model(args.data, args.epochs, args.batch_size, args.lr)
