# tests/integration/reinforcement_learning/run_all_tests.py
"""
Run all RL integration tests and generate a comprehensive report.

This script executes all RL integration tests, collects results, and generates
a detailed report of test coverage and success rates.
"""

import os
import sys
import time
import unittest
import json
from datetime import datetime

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import test modules
from tests.integration.reinforcement_learning.test_improved_llm_rl_bridge import TestImprovedLLMtoRLBridge, TestImprovedLLMtoRLBridgePerformance
from tests.integration.reinforcement_learning.test_llm_rl_continuous_learning import TestContinuousLearningLLMRL
import tests.e2e.rl_module.test_scenarios as scenario_tests

def run_unit_tests():
    """Run unit tests and collect results."""
    print("\n\n=== Running Unit Tests ===\n")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add TestImprovedLLMtoRLBridge tests
    test_suite.addTest(unittest.makeSuite(TestImprovedLLMtoRLBridge))
    
    # Run tests
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    return {
        "total": result.testsRun,
        "passed": result.testsRun - len(result.errors) - len(result.failures),
        "failures": len(result.failures),
        "errors": len(result.errors)
    }

def run_performance_tests():
    """Run performance tests and collect results."""
    print("\n\n=== Running Performance Tests ===\n")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add TestImprovedLLMtoRLBridgePerformance tests
    test_suite.addTest(unittest.makeSuite(TestImprovedLLMtoRLBridgePerformance))
    
    # Run tests
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    return {
        "total": result.testsRun,
        "passed": result.testsRun - len(result.errors) - len(result.failures),
        "failures": len(result.failures),
        "errors": len(result.errors)
    }

def run_learning_tests():
    """Run learning tests and collect results."""
    print("\n\n=== Running Learning Tests ===\n")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add TestContinuousLearningLLMRL tests
    test_suite.addTest(unittest.makeSuite(TestContinuousLearningLLMRL))
    
    # Run tests
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    return {
        "total": result.testsRun,
        "passed": result.testsRun - len(result.errors) - len(result.failures),
        "failures": len(result.failures),
        "errors": len(result.errors)
    }

def run_scenario_tests():
    """Run scenario tests and collect results."""
    print("\n\n=== Running Scenario Tests ===\n")
    
    # Create scenario tester
    tester = scenario_tests.RLScenarioTester()
    
    # Run all scenarios
    results = tester.run_scenarios()
    
    # Print summary
    tester.print_summary()
    
    # Return results with total count
    return {
        "results": results,
        "total": len(results),
        "passed": sum(1 for r in results.values() if r.get("mean_reward", 0) > 0.5),
        "metrics": {scenario: {
            "mean_reward": result.get("mean_reward", 0),
            "completion_rate": result.get("completion_rate", 0)
        } for scenario, result in results.items()}
    }

def generate_report(results):
    """Generate a comprehensive test report."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, f"rl_test_report_{timestamp}.json")
    
    # Calculate overall statistics
    total_tests = (results["unit"]["total"] + results["performance"]["total"] + 
                  results["learning"]["total"] + results["scenarios"]["total"])
    
    passed_tests = (results["unit"]["passed"] + results["performance"]["passed"] + 
                   results["learning"]["passed"] + results["scenarios"]["passed"])
    
    overall_success_rate = 100 * passed_tests / total_tests if total_tests > 0 else 0
    
    # Add summary to results
    results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": overall_success_rate,
        "timestamp": timestamp,
        "test_categories": [
            {"name": "Unit Tests", "tests": results["unit"]["total"], "passed": results["unit"]["passed"]},
            {"name": "Performance Tests", "tests": results["performance"]["total"], "passed": results["performance"]["passed"]},
            {"name": "Learning Tests", "tests": results["learning"]["total"], "passed": results["learning"]["passed"]},
            {"name": "Scenario Tests", "tests": results["scenarios"]["total"], "passed": results["scenarios"]["passed"]}
        ]
    }
    
    # Write report to file
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest report generated: {report_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print(f"TEST SUMMARY ({timestamp})")
    print("="*80)
    print(f"Total Tests:     {total_tests}")
    print(f"Passed Tests:    {passed_tests}")
    print(f"Success Rate:    {overall_success_rate:.2f}%")
    print("-"*80)
    print("Categories:")
    for category in results["summary"]["test_categories"]:
        success_rate = 100 * category["passed"] / category["tests"] if category["tests"] > 0 else 0
        print(f"  {category['name']}: {category['passed']}/{category['tests']} ({success_rate:.2f}%)")
    print("="*80)
    
    return report_file

if __name__ == "__main__":
    start_time = time.time()
    
    # Run all tests
    results = {
        "unit": run_unit_tests(),
        "performance": run_performance_tests(),
        "learning": run_learning_tests(),
        "scenarios": run_scenario_tests()
    }
    
    # Generate report
    report_file = generate_report(results)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")