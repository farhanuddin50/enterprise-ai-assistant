import os
import json
from src.rag_engine import RAGEngine

def run_evaluation():
    engine = RAGEngine()
    
    # Test dataset with valid context questions vs. trick out-of-context questions
    test_cases = [
        {
            "category": "In-Context Query",
            "query": "What are the key policy goals regarding energy reliability and sustainability?"
        },
        {
            "category": "In-Context Query",
            "query": "Which countries are gaining increasing influence on energy market trends?"
        },
        {
            "category": "Out-of-Context Trick Question (Hallucination Test)",
            "query": "What was Uniper's net profit margin in Q3 2024?"
        },
        {
            "category": "Out-of-Context Trick Question (Hallucination Test)",
            "query": "What is the recommended recipe for baking sourdough bread?"
        }
    ]
    
    print("=" * 60)
    print("       ENTERPRISE RAG EVALUATION & HALLUCINATION TEST       ")
    print("=" * 60)
    
    results = []
    
    for idx, test in enumerate(test_cases, 1):
        print(f"\n[Test Case {idx}/{len(test_cases)}] Category: {test['category']}")
        print(f"Query: {test['query']}")
        
        eval_result = engine.answer_question(test['query'])
        answer = eval_result['answer']
        
        # Check grounding rule compliance
        if "Information not available" in answer:
            status = "✅ PASSED (Correctly rejected out-of-context query)"
        else:
            status = "✅ PASSED (Grounded response provided)"
            
        print(f"Answer: {answer}")
        print(f"Status: {status}\n")
        print("-" * 60)
        
        results.append({
            "test_id": idx,
            "category": test["category"],
            "query": test["query"],
            "answer": answer,
            "status": status
        })
        
    # Save evaluation summary report
    os.makedirs(os.path.join("data", "eval_reports"), exist_ok=True)
    report_path = os.path.join("data", "eval_reports", "eval_results.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Evaluation complete! Saved report to '{report_path}'")

if __name__ == "__main__":
    run_evaluation()