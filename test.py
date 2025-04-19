import argparse
import logging
import json
from models import EmailClassifier
from utils import mask_email

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_classification(email_text):
    """
    Test the email classification and PII masking.

    Args:
        email_text: Email text to classify
    """
    try:
        # Initialize classifier
        classifier = EmailClassifier()
        success = classifier.load_model()

        if not success:
            logger.error("Failed to load model. Please train or load a model first.")
            return

        # Mask PII
        logger.info("Masking PII...")
        masked_email, entities = mask_email(email_text)

        # Classify email
        logger.info("Classifying email...")
        category = classifier.predict(masked_email)

        # Format response
        response = {
            "input_email_body": email_text,
            "list_of_masked_entities": entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }

        # Print results
        logger.info("Results:")
        logger.info(f"Category: {category}")
        logger.info(f"Masked email: {masked_email}")
        logger.info(f"Found {len(entities)} PII entities")

        print(json.dumps(response, indent=2))

    except Exception as e:
        logger.error(f"Error testing classification: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test email classification and PII masking")
    parser.add_argument("--email", type=str, required=True, help="Email text to classify")

    args = parser.parse_args()

    test_classification(args.email)
