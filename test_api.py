#!/usr/bin/env python3
"""
Quick API Test - Test your Capital Structure Optimizer API
Run this while optimizer_app.py is running in another terminal
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:5000"

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_model_status():
    """Test if model is loaded"""
    print_header("TEST 1: Check Model Status")
    
    try:
        response = requests.get(f"{BASE_URL}/api/model-status")
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Model Loaded: {data.get('loaded', False)}")
        print(f"Model Path: {data.get('model_path', 'N/A')}")
        
        if data.get('loaded'):
            print("✅ Model Status: SUCCESS")
            return True
        else:
            print("❌ Model Status: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_examples():
    """Get example companies"""
    print_header("TEST 2: Get Example Companies")
    
    try:
        response = requests.get(f"{BASE_URL}/api/example-companies")
        examples = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Examples Available: {list(examples.keys())}")
        
        for key, example in examples.items():
            print(f"\n  {example.get('name', 'Unknown')}:")
            print(f"    - Ticker: {example.get('ticker')}")
            print(f"    - Debt: ${example.get('debt')}M")
            print(f"    - Equity: ${example.get('equity')}M")
        
        print("\n✅ Examples: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_optimization(company_data, name):
    """Run optimization on company data"""
    print_header(f"TEST 3: Optimize - {name}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/optimize",
            json=company_data,
            headers={"Content-Type": "application/json"}
        )
        
        result = response.json()
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200 and result.get('success'):
            print(f"Company: {company_data.get('ticker')}")
            print(f"Initial Leverage: {company_data['debt']/(company_data['debt']+company_data['equity'])*100:.1f}%")
            
            if result.get('results'):
                final_result = result['results'][-1]
                print(f"Final Leverage: {final_result.get('leverage', 0)*100:.1f}%")
                print(f"Final Reward: {final_result.get('reward', 0):.4f}")
                print(f"Total Reward: {sum(r.get('reward', 0) for r in result['results']):.4f}")
            
            print("✅ Optimization: SUCCESS")
            return True
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║         🧪 Capital Structure Optimizer - API Test Suite         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Make sure optimizer_app.py is running:
  $ python optimizer_app.py

This script tests all API endpoints.
    """)
    
    print(f"\nConnecting to: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Model Status
    model_ok = test_model_status()
    time.sleep(1)
    
    if not model_ok:
        print("\n❌ Model not loaded. Cannot continue tests.")
        return
    
    # Test 2: Examples
    examples_ok = test_examples()
    time.sleep(1)
    
    # Test 3: Optimization - Tech Startup
    tech_startup = {
        'ticker': 'TECH',
        'debt': 50,
        'equity': 300,
        'cash': 100,
        'cf_history': [0.05, 0.08, 0.10, 0.12, 0.15],
        'steps': 5
    }
    tech_ok = test_optimization(tech_startup, "Tech Startup")
    time.sleep(1)
    
    # Test 4: Optimization - Mature Company
    mature = {
        'ticker': 'MAT',
        'debt': 200,
        'equity': 400,
        'cash': 50,
        'cf_history': [0.20, 0.20, 0.20, 0.20, 0.20],
        'steps': 5
    }
    mature_ok = test_optimization(mature, "Mature Company")
    time.sleep(1)
    
    # Test 5: Optimization - Distressed
    distressed = {
        'ticker': 'DIST',
        'debt': 300,
        'equity': 100,
        'cash': 10,
        'cf_history': [0.05, 0.03, 0.02, -0.01, 0.01],
        'steps': 5
    }
    distressed_ok = test_optimization(distressed, "Distressed Company")
    
    # Summary
    print_header("TEST SUMMARY")
    
    tests = {
        'Model Status': model_ok,
        'Examples': examples_ok,
        'Tech Startup': tech_ok,
        'Mature Company': mature_ok,
        'Distressed Company': distressed_ok
    }
    
    passed = sum(1 for v in tests.values() if v)
    total = len(tests)
    
    for test_name, result in tests.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your API is working correctly!")
        print("\n📊 Next Steps:")
        print("  1. Open: http://localhost:5000")
        print("  2. Try the web interface")
        print("  3. Load example companies")
        print("  4. Enter custom company data")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")
    
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
