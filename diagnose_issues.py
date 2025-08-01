#!/usr/bin/env python3
"""
Diagnostic script for Enhanced Document Query API
Helps identify and resolve common configuration and connection issues
"""

import os
import asyncio
import logging
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_gemini_connection():
    """Test Gemini API connection and configuration"""
    logger.info("🔍 Testing Gemini API connection...")
    
    try:
        import google.generativeai as genai
        from config import config
        
        if not config.GEMINI_API_KEY:
            logger.error("❌ GEMINI_API_KEY not found in environment")
            return False
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(config.GEMINI_CHAT_MODEL)
        
        # Test simple generation
        response = model.generate_content("Hello, respond with just 'OK'")
        if response and response.text:
            logger.info("✅ Gemini API connection successful")
            logger.info(f"   Model: {config.GEMINI_CHAT_MODEL}")
            logger.info(f"   Response: {response.text.strip()}")
            return True
        else:
            logger.error("❌ Gemini API returned empty response")
            return False
            
    except Exception as e:
        logger.error(f"❌ Gemini API connection failed: {e}")
        return False

async def test_pinecone_connection():
    """Test Pinecone connection and index configuration"""
    logger.info("🔍 Testing Pinecone connection...")
    
    try:
        from pinecone import Pinecone
        from config import config
        
        if not config.PINECONE_API_KEY:
            logger.error("❌ PINECONE_API_KEY not found in environment")
            return False
        
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        
        # Check if index exists
        if not pc.has_index(config.PINECONE_INDEX_NAME):
            logger.warning(f"⚠️ Pinecone index '{config.PINECONE_INDEX_NAME}' does not exist")
            logger.info("   You may need to create the index first")
            return False
        
        # Connect to index
        index = pc.Index(config.PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        
        logger.info("✅ Pinecone connection successful")
        logger.info(f"   Index: {config.PINECONE_INDEX_NAME}")
        logger.info(f"   Dimension: {config.PINECONE_DIMENSION}")
        logger.info(f"   Total vectors: {stats.total_vector_count}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pinecone connection failed: {e}")
        return False

async def test_database_connection():
    """Test PostgreSQL database connection"""
    logger.info("🔍 Testing PostgreSQL database connection...")
    
    try:
        from database_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Test connection
        success = await db_manager.test_connection()
        if success:
            logger.info("✅ PostgreSQL database connection successful")
            return True
        else:
            logger.error("❌ PostgreSQL database connection test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ PostgreSQL database connection failed: {e}")
        logger.info("   This is optional - the system can work without PostgreSQL")
        return False

def test_redis_connection():
    """Test Redis connection"""
    logger.info("🔍 Testing Redis connection...")
    
    try:
        import redis
        from config import config
        
        r = redis.from_url(config.REDIS_URL)
        r.ping()
        
        logger.info("✅ Redis connection successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        logger.info("   This is optional - the system can work without Redis caching")
        return False

def check_environment_variables():
    """Check required environment variables"""
    logger.info("🔍 Checking environment variables...")
    
    required_vars = [
        'GEMINI_API_KEY',
        'PINECONE_API_KEY',
        'PINECONE_INDEX_NAME'
    ]
    
    optional_vars = [
        'DATABASE_URL',
        'REDIS_URL',
        'GOOGLE_CLOUD_PROJECT'
    ]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_required:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_required)}")
        return False
    else:
        logger.info("✅ All required environment variables are set")
    
    if missing_optional:
        logger.warning(f"⚠️ Missing optional environment variables: {', '.join(missing_optional)}")
    
    return True

def check_dimension_compatibility():
    """Check if Pinecone dimension matches Gemini embedding dimension"""
    logger.info("🔍 Checking dimension compatibility...")
    
    try:
        from config import config
        
        gemini_dim = 768  # Gemini text-embedding-004 dimension
        pinecone_dim = config.PINECONE_DIMENSION
        
        if pinecone_dim != gemini_dim:
            logger.warning(f"⚠️ Dimension mismatch:")
            logger.warning(f"   Gemini embeddings: {gemini_dim} dimensions")
            logger.warning(f"   Pinecone index: {pinecone_dim} dimensions")
            logger.warning("   The system will pad/truncate embeddings, but this may affect performance")
            return False
        else:
            logger.info("✅ Dimensions are compatible")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error checking dimensions: {e}")
        return False

async def run_diagnostics():
    """Run all diagnostic tests"""
    logger.info("🚀 Starting Enhanced Document Query API Diagnostics")
    logger.info("=" * 60)
    
    results = {}
    
    # Check environment variables first
    results['env_vars'] = check_environment_variables()
    
    # Check dimension compatibility
    results['dimensions'] = check_dimension_compatibility()
    
    # Test connections
    results['gemini'] = await test_gemini_connection()
    results['pinecone'] = await test_pinecone_connection()
    results['database'] = await test_database_connection()
    results['redis'] = test_redis_connection()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)
    
    critical_issues = []
    warnings = []
    
    if not results['env_vars']:
        critical_issues.append("Missing required environment variables")
    
    if not results['gemini']:
        critical_issues.append("Gemini API connection failed")
    
    if not results['pinecone']:
        critical_issues.append("Pinecone connection failed")
    
    if not results['dimensions']:
        warnings.append("Dimension mismatch between Gemini and Pinecone")
    
    if not results['database']:
        warnings.append("PostgreSQL database not available (optional)")
    
    if not results['redis']:
        warnings.append("Redis caching not available (optional)")
    
    if critical_issues:
        logger.error("❌ CRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            logger.error(f"   • {issue}")
        logger.error("\n   The API will not work properly until these are resolved.")
    else:
        logger.info("✅ All critical components are working!")
    
    if warnings:
        logger.warning("⚠️ WARNINGS:")
        for warning in warnings:
            logger.warning(f"   • {warning}")
    
    logger.info("=" * 60)
    
    if critical_issues:
        logger.info("🔧 RECOMMENDED ACTIONS:")
        logger.info("1. Check your .env file and ensure all required API keys are set")
        logger.info("2. Verify your Pinecone index exists and is accessible")
        logger.info("3. Test your Gemini API key at https://makersuite.google.com/")
        logger.info("4. Check the logs above for specific error details")
    else:
        logger.info("🎉 System is ready to use!")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
