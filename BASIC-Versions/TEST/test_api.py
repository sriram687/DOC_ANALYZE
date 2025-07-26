# test_api.py - Test examples for the Document Query API

import asyncio
import aiohttp
import json
from pathlib import Path

class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def test_health_check(self):
        """Test the health endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                result = await response.json()
                print("Health Check:", result)
                return response.status == 200
    
    async def test_document_query(self, query, file_path):
        """Test document query with file upload"""
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('query', query)
            
            with open(file_path, 'rb') as f:
                data.add_field('file', f, filename=Path(file_path).name)
                
                async with session.post(f"{self.base_url}/ask-document", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error = await response.text()
                        print(f"Error {response.status}: {error}")
                        return None

# Example test scenarios
async def run_tests():
    tester = APITester()
    
    # Test health check
    print("=== Testing Health Check ===")
    await tester.test_health_check()
    
    # Example queries for different document types
    test_cases = [
        {
            "query": "Does this policy cover knee surgery, and what are the conditions?",
            "description": "Insurance coverage query",
            "expected_intent": "coverage_check"
        },
        {
            "query": "What are the employee benefits regarding maternity leave?",
            "description": "HR policy query",
            "expected_intent": "policy_lookup"
        },
        {
            "query": "Are there any compliance requirements for data handling?",
            "description": "Compliance query",
            "expected_intent": "compliance_check"
        },
        {
            "query": "What is the cancellation policy for this service?",
            "description": "General policy query",
            "expected_intent": "policy_lookup"
        }
    ]
    
    # NOTE: You'll need to provide actual test documents
    # test_file = "sample_insurance_policy.pdf"
    # 
    # for test_case in test_cases:
    #     print(f"\n=== Testing: {test_case['description']} ===")
    #     print(f"Query: {test_case['query']}")
    #     
    #     result = await tester.test_document_query(test_case['query'], test_file)
    #     if result:
    #         print(f"Answer: {result['answer']}")
    #         print(f"Confidence: {result['confidence']}")
    #         print(f"Conditions: {result['conditions']}")
    #         print(f"Evidence count: {len(result['evidence'])}")
    #         print(f"Processing time: {result['processing_time']:.2f}s")

# Sample test document creator (for testing purposes)
def create_sample_documents():
    """Create sample documents for testing"""
    
    # Sample insurance policy text
    insurance_policy = """
    COMPREHENSIVE HEALTH INSURANCE POLICY
    
    Section 1: Coverage Details
    This policy provides comprehensive health coverage for the insured and eligible dependents.
    
    Section 2: Surgical Procedures
    The plan covers orthopedic procedures including but not limited to:
    - Knee surgery and replacement
    - Hip replacement
    - Shoulder procedures
    
    All surgical procedures require pre-authorization from the insurance provider.
    Medical reports and physician recommendations must be submitted 30 days prior to surgery.
    
    Section 3: Pre-Authorization Requirements
    For knee surgery specifically:
    1. Radiology report showing medical necessity
    2. Orthopedic surgeon's recommendation
    3. Failed conservative treatment documentation
    4. Physical therapy records (minimum 6 weeks)
    
    Section 4: Exclusions
    The following are not covered:
    - Cosmetic procedures
    - Experimental treatments
    - Pre-existing conditions (first 12 months)
    """
    
    with open("sample_insurance_policy.txt", "w") as f:
        f.write(insurance_policy)
    
    # Sample HR policy
    hr_policy = """
    EMPLOYEE HANDBOOK
    
    Chapter 5: Leave Policies
    
    5.1 Maternity Leave
    Eligible employees are entitled to up to 12 weeks of maternity leave.
    
    Benefits during maternity leave:
    - Full salary for first 6 weeks
    - 60% salary for weeks 7-12
    - Continued health insurance coverage
    - Job protection guarantee
    
    Eligibility requirements:
    - Minimum 12 months of employment
    - Full-time status (30+ hours per week)
    - Advance notice of 30 days when possible
    
    5.2 Paternity Leave
    Male employees are entitled to 2 weeks paid paternity leave.
    
    5.3 Family Medical Leave
    Additional unpaid leave may be available under FMLA guidelines.
    """
    
    with open("sample_hr_policy.txt", "w") as f:
        f.write(hr_policy)
    
    # Sample compliance document
    compliance_doc = """
    DATA HANDLING AND PRIVACY COMPLIANCE MANUAL
    
    Section 1: Data Classification
    All data must be classified according to sensitivity levels:
    - Public: No restrictions
    - Internal: Company personnel only
    - Confidential: Authorized personnel only
    - Restricted: Executive approval required
    
    Section 2: Data Handling Requirements
    
    2.1 Personal Data Processing
    All personal data must be:
    - Processed lawfully and transparently
    - Collected for specified purposes only
    - Adequate, relevant, and limited
    - Accurate and up-to-date
    - Retained only as long as necessary
    - Processed securely
    
    2.2 Technical Safeguards
    Required security measures include:
    - Encryption of data at rest and in transit
    - Multi-factor authentication for system access
    - Regular security audits and penetration testing
    - Incident response procedures
    - Staff training on data protection
    
    2.3 Compliance Monitoring
    Monthly compliance reviews are mandatory.
    Annual third-party security assessments required.
    Breach notification within 72 hours to regulatory authorities.
    """
    
    with open("sample_compliance_doc.txt", "w") as f:
        f.write(compliance_doc)
    
    print("Sample documents created:")
    print("- sample_insurance_policy.txt")
    print("- sample_hr_policy.txt") 
    print("- sample_compliance_doc.txt")

# Integration test with actual file
async def integration_test():
    """Run integration tests with sample documents"""
    
    # Create sample documents
    create_sample_documents()
    
    tester = APITester()
    
    test_scenarios = [
        {
            "file": "sample_insurance_policy.txt",
            "queries": [
                "Does this policy cover knee surgery?",
                "What are the pre-authorization requirements for knee surgery?",
                "What procedures are excluded from coverage?",
                "How much advance notice is required for surgery?"
            ]
        },
        {
            "file": "sample_hr_policy.txt", 
            "queries": [
                "What are the maternity leave benefits?",
                "How long do I need to work before I'm eligible for maternity leave?",
                "Is paternity leave available and for how long?",
                "What percentage of salary is paid during maternity leave?"
            ]
        },
        {
            "file": "sample_compliance_doc.txt",
            "queries": [
                "What are the data classification levels?",
                "What security measures are required for personal data?",
                "How often should compliance reviews be conducted?",
                "What is the breach notification timeline?"
            ]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"Testing with: {scenario['file']}")
        print('='*60)
        
        for query in scenario['queries']:
            print(f"\nQuery: {query}")
            print("-" * 50)
            
            try:
                result = await tester.test_document_query(query, scenario['file'])
                if result:
                    print(f"Answer: {result['answer']}")
                    print(f"Confidence: {result['confidence']:.2f}")
                    
                    if result['conditions']:
                        print("Conditions:")
                        for condition in result['conditions']:
                            print(f"  • {condition}")
                    
                    if result['evidence']:
                        print("Evidence:")
                        for i, evidence in enumerate(result['evidence'][:2], 1):  # Show first 2
                            print(f"  {i}. {evidence.get('text', 'N/A')[:100]}...")
                    
                    print(f"Processing time: {result['processing_time']:.2f}s")
                else:
                    print("❌ Query failed")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")

# Performance testing
async def performance_test():
    """Test API performance with concurrent requests"""
    import time
    
    tester = APITester()
    create_sample_documents()
    
    queries = [
        "Does this policy cover knee surgery?",
        "What are the pre-authorization requirements?", 
        "What procedures are excluded?",
        "What are the eligibility requirements?"
    ]
    
    # Test concurrent requests
    print("\n=== Performance Test: Concurrent Requests ===")
    
    async def single_request(query):
        start = time.time()
        result = await tester.test_document_query(query, "sample_insurance_policy.txt")
        end = time.time()
        return end - start, result is not None
    
    # Run 4 concurrent requests
    start_time = time.time()
    tasks = [single_request(query) for query in queries]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    successful_requests = sum(1 for _, success in results if success)
    avg_response_time = sum(time for time, _ in results) / len(results)
    
    print(f"Total requests: {len(queries)}")
    print(f"Successful requests: {successful_requests}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average response time: {avg_response_time:.2f}s")
    print(f"Requests per second: {len(queries) / total_time:.2f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        
        if test_type == "integration":
            asyncio.run(integration_test())
        elif test_type == "performance":
            asyncio.run(performance_test())
        elif test_type == "create-samples":
            create_sample_documents()
        else:
            print("Usage: python test_api.py [integration|performance|create-samples]")
    else:
        print("Running basic tests...")
        asyncio.run(run_tests())