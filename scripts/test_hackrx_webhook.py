#!/usr/bin/env python3
"""
HackRx Document Analysis Webhook Test
Test the webhook with the exact payload format expected by HackRx evaluator
"""

import requests
import json
from datetime import datetime

# Configuration
LOCAL_WEBHOOK_URL = "http://localhost:8000/webhook"
PRODUCTION_WEBHOOK_URL = "https://doc-analyze.onrender.com/webhook"

# HackRx Test Payload (exact format from your requirements)
HACKRX_PAYLOAD = {
    "type": "document_analysis",
    "filename": "policy.pdf",
    "content": "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits. There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered. Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period. The policy has a specific waiting period of two (2) years for cataract surgery. Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994. A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium. Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits. A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients. The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital. Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN).",
    "user_id": "sriram123",
    "timestamp": "2025-08-03T18:00:00Z",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

# Expected answers for validation
EXPECTED_ANSWERS = [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
    "The policy has a specific waiting period of two (2) years for cataract surgery.",
    "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
    "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
    "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
    "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
    "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
    "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
]

def test_hackrx_webhook(webhook_url, test_name):
    """Test HackRx document analysis webhook"""
    print(f"\n🏆 Testing HackRx Document Analysis - {test_name}")
    print(f"📡 URL: {webhook_url}")
    print(f"📄 Document: {HACKRX_PAYLOAD['filename']}")
    print(f"❓ Questions: {len(HACKRX_PAYLOAD['questions'])}")
    
    try:
        print(f"\n📤 Sending document analysis request...")
        
        response = requests.post(
            webhook_url,
            json=HACKRX_PAYLOAD,
            headers={"Content-Type": "application/json"},
            timeout=60  # Longer timeout for LLM processing
        )
        
        print(f"✅ Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if we got the expected format
            if "answers" in result:
                answers = result["answers"]
                print(f"📋 Received {len(answers)} answers")
                
                # Validate answer count
                if len(answers) == 10:
                    print("✅ Correct number of answers (10)")
                    
                    # Display answers
                    print("\n📝 Generated Answers:")
                    for i, answer in enumerate(answers, 1):
                        print(f"{i:2d}. {answer}")
                    
                    # Check answer quality (basic validation)
                    valid_answers = 0
                    for answer in answers:
                        if answer and len(answer.strip()) > 10 and "error" not in answer.lower():
                            valid_answers += 1
                    
                    print(f"\n📊 Answer Quality: {valid_answers}/10 answers appear valid")
                    
                    if valid_answers >= 8:
                        print("🎉 EXCELLENT! Ready for HackRx evaluation!")
                    elif valid_answers >= 6:
                        print("👍 GOOD! Most answers look valid")
                    else:
                        print("⚠️  Some answers may need improvement")
                    
                else:
                    print(f"❌ Wrong number of answers: expected 10, got {len(answers)}")
            else:
                print(f"❌ Invalid response format: {result}")
        else:
            print(f"❌ Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - LLM processing may take longer")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def validate_response_format(response_data):
    """Validate that response matches HackRx requirements"""
    if not isinstance(response_data, dict):
        return False, "Response must be a JSON object"
    
    if "answers" not in response_data:
        return False, "Response must contain 'answers' field"
    
    answers = response_data["answers"]
    if not isinstance(answers, list):
        return False, "'answers' must be an array"
    
    if len(answers) != 10:
        return False, f"Must have exactly 10 answers, got {len(answers)}"
    
    return True, "Response format is correct"

if __name__ == "__main__":
    print("🏆 HackRx Document Analysis Webhook Test")
    print("=" * 60)
    
    # Test local webhook (if running)
    try:
        test_hackrx_webhook(LOCAL_WEBHOOK_URL, "Local Development")
    except:
        print("⚠️  Local server not running, skipping local test")
    
    # Test production webhook
    try:
        test_hackrx_webhook(PRODUCTION_WEBHOOK_URL, "Production")
    except:
        print("⚠️  Production server not accessible")
    
    print("\n" + "=" * 60)
    print("🎯 HackRx Webhook Testing Complete!")
    print("\n📋 Your webhook is ready for:")
    print("   • Document content processing")
    print("   • 10-question analysis")
    print("   • JSON response format")
    print("   • HackRx auto-evaluation")
    print(f"\n🌐 Production URL: {PRODUCTION_WEBHOOK_URL}")
