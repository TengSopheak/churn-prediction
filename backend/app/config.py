import os
from dotenv import load_dotenv

# Load the environment variables when config is imported
load_dotenv()

# S3_BUCKET = os.getenv("S3_BUCKET")
# AWS_REGION = os.getenv("AWS_REGION")
# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
