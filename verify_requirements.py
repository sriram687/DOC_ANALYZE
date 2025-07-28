#!/usr/bin/env python3
"""
Verify that all required dependencies are installed and working
"""
import sys

def test_imports():
    """Test importing all core dependencies"""
    tests = [
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("Google AI", "google.generativeai"),
        ("Pinecone", "pinecone"),
        ("SQLAlchemy", "sqlalchemy"),
        ("AsyncPG", "asyncpg"),
        ("Psycopg", "psycopg"),
        ("NumPy", "numpy"),
        ("PyMuPDF", "fitz"),
        ("Python-DOCX", "docx"),
        ("Pydantic", "pydantic"),
        ("Python-dotenv", "dotenv"),
        ("Requests", "requests"),
    ]
    
    print("🔍 Verifying core dependencies...")
    print("=" * 50)
    
    failed = []
    for name, module in tests:
        try:
            __import__(module)
            print(f"✅ {name:<20} - OK")
        except ImportError as e:
            print(f"❌ {name:<20} - FAILED: {e}")
            failed.append(name)
    
    print("=" * 50)
    
    if failed:
        print(f"❌ {len(failed)} dependencies failed to import:")
        for name in failed:
            print(f"   - {name}")
        return False
    else:
        print(f"✅ All {len(tests)} core dependencies are working!")
        return True

def test_optional_imports():
    """Test optional dependencies"""
    optional_tests = [
        ("Sentence Transformers", "sentence_transformers"),
        ("Redis", "redis"),
    ]
    
    print("\n🔍 Checking optional dependencies...")
    print("=" * 50)
    
    for name, module in optional_tests:
        try:
            __import__(module)
            print(f"✅ {name:<20} - Available")
        except ImportError:
            print(f"⚠️  {name:<20} - Not installed (optional)")
    
    print("=" * 50)

if __name__ == "__main__":
    print("🚀 Enhanced Document Query API - Dependency Verification")
    print(f"🐍 Python version: {sys.version}")
    print()
    
    success = test_imports()
    test_optional_imports()
    
    if success:
        print("\n🎉 All core dependencies verified successfully!")
        print("📚 Ready to run: python main.py")
    else:
        print("\n❌ Some dependencies are missing. Please run:")
        print("📦 pip install -r requirements.txt")
        sys.exit(1)
