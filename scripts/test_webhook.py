#!/usr/bin/env python3
"""
Webhook Testing Script for Document Analyzer API
Test your webhook endpoint locally and in production
"""

import requests
import json
from datetime import datetime

# Configuration
LOCAL_WEBHOOK_URL = "http://localhost:8000/webhook"
PRODUCTION_WEBHOOK_URL = "https://doc-analyze.onrender.com/webhook"

def test_webhook(webhook_url, test_name):
    """Test webhook with sample data"""
    print(f"\n🧪 Testing {test_name}")
    print(f"📡 URL: {webhook_url}")
    
    # Sample webhook payloads
    test_payloads = [
        {
            "type": "test",
            "message": "Hello webhook!",
            "timestamp": datetime.now().isoformat()
        },
        {
            "type": "payment.success",
            "payment_id": "pay_123456",
            "amount": 29.99,
            "currency": "USD",
            "customer_email": "user@example.com"
        },
        {
            "type": "document.uploaded",
            "document_id": "doc_789",
            "filename": "contract.pdf",
            "size": 1024000,
            "user_id": "user_456"
        }
    ]
    
    for i, payload in enumerate(test_payloads, 1):
        try:
            print(f"\n📤 Test {i}: {payload['type']}")
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            print(f"✅ Status: {response.status_code}")
            print(f"📋 Response: {response.json()}")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def test_webhook_info(base_url):
    """Test webhook info endpoint"""
    try:
        info_url = f"{base_url}/webhook/test"
        response = requests.get(info_url)
        print(f"\n📋 Webhook Info:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Error getting webhook info: {e}")

if __name__ == "__main__":
    print("🔗 Webhook Testing Script")
    print("=" * 50)
    
    # Test local webhook (if running)
    try:
        test_webhook_info("http://localhost:8000")
        test_webhook(LOCAL_WEBHOOK_URL, "Local Development")
    except:
        print("⚠️  Local server not running, skipping local tests")
    
    # Test production webhook
    try:
        test_webhook_info("https://doc-analyze.onrender.com")
        test_webhook(PRODUCTION_WEBHOOK_URL, "Production")
    except:
        print("⚠️  Production server not accessible")
    
    print("\n✅ Webhook testing completed!")
    print("\n📝 Your webhook URL is ready:")
    print("🌐 https://doc-analyze.onrender.com/webhook")
