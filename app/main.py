from pipeline import RAGPipeline
from evaluation import RAGEvaluator

pipeline=RAGPipeline()
evaluator=RAGEvaluator(pipeline)
#query=input("how can i help you? \n")
#print(pipeline.ask(query))
test_cases = [
    {
        "question": "Who invented the telephone?",
        "expected_chunk_id": 0,
        "expected_keywords": ["Alexander Graham Bell"]
    },
    {
        "question": "In what year was the telephone patented?",
        "expected_chunk_id": 0,
        "expected_keywords": ["1876"]
    },
    {
        "question": "Where were the world's first definitive tests of the telephone conducted?",
        "expected_chunk_id": 12,
        "expected_keywords": ["Brantford", "1876"]
    },
    {
        "question": "When was the Bell Telephone Company of Canada incorporated?",
        "expected_chunk_id": 16,
        "expected_keywords": ["1880"]
    },
    {
        "question": "What major decision did the CRTC make in 1992 regarding telecommunications?",
        "expected_chunk_id": 62,
        "expected_keywords": ["1992", "competitive entry", "long distance"]
    }
]
eval_output=evaluator.run(test_cases, use_rerank=False)   
print(f"For use rerank false \n {eval_output["metrics"]}")

eval_output2=evaluator.run(test_cases, use_rerank=True)   
print(f"For use rerank true \n {eval_output2["metrics"]}")