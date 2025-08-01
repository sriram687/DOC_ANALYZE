#!/usr/bin/env python3
"""
Fix Pinecone dimension mismatch issue
This script helps create a new Pinecone index with the correct dimensions for Gemini embeddings
"""

import os
import logging
import time
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_new_pinecone_index():
    """Create a new Pinecone index with correct dimensions for Gemini embeddings"""
    
    try:
        from pinecone import Pinecone, ServerlessSpec
        from config import config
        
        if not config.PINECONE_API_KEY:
            logger.error("❌ PINECONE_API_KEY not found in environment")
            return False
        
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        
        # Gemini text-embedding-004 uses 768 dimensions
        correct_dimension = 768
        new_index_name = f"{config.PINECONE_INDEX_NAME}-gemini"
        
        logger.info(f"🔧 Creating new Pinecone index: {new_index_name}")
        logger.info(f"   Dimension: {correct_dimension}")
        logger.info(f"   Metric: {config.PINECONE_METRIC}")
        
        # Check if index already exists
        if pc.has_index(new_index_name):
            logger.warning(f"⚠️ Index '{new_index_name}' already exists")
            
            # Get index stats
            index = pc.Index(new_index_name)
            stats = index.describe_index_stats()
            logger.info(f"   Current vectors: {stats.total_vector_count}")
            
            response = input("Do you want to delete and recreate it? (y/N): ")
            if response.lower() != 'y':
                logger.info("Keeping existing index")
                return True
            
            logger.info("🗑️ Deleting existing index...")
            pc.delete_index(new_index_name)
            time.sleep(10)  # Wait for deletion to complete
        
        # Create new index
        logger.info("🏗️ Creating new index...")
        pc.create_index(
            name=new_index_name,
            dimension=correct_dimension,
            metric=config.PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region=config.PINECONE_ENVIRONMENT or "us-east-1"
            )
        )
        
        # Wait for index to be ready
        logger.info("⏳ Waiting for index to be ready...")
        time.sleep(30)
        
        # Verify index
        if pc.has_index(new_index_name):
            index = pc.Index(new_index_name)
            stats = index.describe_index_stats()
            logger.info("✅ Index created successfully!")
            logger.info(f"   Name: {new_index_name}")
            logger.info(f"   Dimension: {correct_dimension}")
            logger.info(f"   Status: Ready")
            
            # Update .env file suggestion
            logger.info("🔧 To use this new index, update your .env file:")
            logger.info(f"   PINECONE_INDEX_NAME={new_index_name}")
            logger.info(f"   PINECONE_DIMENSION={correct_dimension}")
            
            return True
        else:
            logger.error("❌ Index creation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error creating Pinecone index: {e}")
        return False

def update_env_file(new_index_name: str):
    """Update .env file with new index configuration"""
    
    try:
        # Read current .env file
        with open('.env', 'r') as f:
            lines = f.readlines()
        
        # Update relevant lines
        updated_lines = []
        for line in lines:
            if line.startswith('PINECONE_INDEX_NAME='):
                updated_lines.append(f'PINECONE_INDEX_NAME={new_index_name}\n')
            elif line.startswith('PINECONE_DIMENSION='):
                updated_lines.append('PINECONE_DIMENSION=768\n')
            else:
                updated_lines.append(line)
        
        # Write updated .env file
        with open('.env', 'w') as f:
            f.writelines(updated_lines)
        
        logger.info("✅ Updated .env file with new configuration")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating .env file: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Pinecone Dimension Fix Tool")
    logger.info("=" * 50)
    
    logger.info("This tool will help fix the dimension mismatch between")
    logger.info("Gemini embeddings (768 dimensions) and your Pinecone index.")
    logger.info("")
    
    # Check current configuration
    try:
        from config import config
        logger.info(f"Current Pinecone index: {config.PINECONE_INDEX_NAME}")
        logger.info(f"Current dimension: {config.PINECONE_DIMENSION}")
        logger.info(f"Required dimension: 768 (for Gemini)")
        logger.info("")
        
        if config.PINECONE_DIMENSION == 768:
            logger.info("✅ Dimensions are already correct!")
            return
        
    except Exception as e:
        logger.error(f"❌ Error reading configuration: {e}")
        return
    
    # Ask user for confirmation
    print("This will create a new Pinecone index with the correct dimensions.")
    print("Your existing index will not be affected.")
    print("")
    response = input("Do you want to proceed? (y/N): ")
    
    if response.lower() != 'y':
        logger.info("Operation cancelled")
        return
    
    # Create new index
    success = create_new_pinecone_index()
    
    if success:
        new_index_name = f"{config.PINECONE_INDEX_NAME}-gemini"
        
        # Ask if user wants to update .env file
        response = input("\nDo you want to automatically update your .env file? (y/N): ")
        if response.lower() == 'y':
            update_env_file(new_index_name)
        else:
            logger.info("Please manually update your .env file:")
            logger.info(f"   PINECONE_INDEX_NAME={new_index_name}")
            logger.info("   PINECONE_DIMENSION=768")
        
        logger.info("")
        logger.info("🎉 Fix completed successfully!")
        logger.info("You can now restart your application.")
    else:
        logger.error("❌ Fix failed. Please check the errors above.")

if __name__ == "__main__":
    main()
