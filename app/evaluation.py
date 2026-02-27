from pipeline import RAGPipeline
import time


class RAGEvaluator:
    
    def __init__(self,pipeline):
        self.pipeline=pipeline

    def run(self,test_cases,use_rerank=True):
        results=[]
        final_result={}

        for test_case in test_cases:
            
            question=test_case["question"]
            retrieval_start = time.perf_counter()
            if  use_rerank:
                retrieved=self.pipeline.search_with_rerank(question)
            else:
                retrieved=self.pipeline.search(question)
            retrieval_time = time.perf_counter() - retrieval_start

            generation_start = time.perf_counter()

            answer = self.pipeline.generate_answer_from_context(question, retrieved)

            generation_time = time.perf_counter() - generation_start
            total_time = retrieval_time + generation_time

            retreived_chunk_ids=[chunk["chunk_id"] for chunk in retrieved]
            if len(retreived_chunk_ids)==0:
                top1_correct=False
                topk_correct=False
            else:
                top1_correct=(test_case["expected_chunk_id"]==retreived_chunk_ids[0])
                topk_correct=(test_case["expected_chunk_id"] in retreived_chunk_ids) 
            if test_case["expected_chunk_id"] not in retreived_chunk_ids:
                reciprocal_rank=0
            else:
                reciprocal_rank=1/(1+retreived_chunk_ids.index(test_case["expected_chunk_id"]))

            answer_correct=all(exp_key.lower() in answer.lower() for exp_key in test_case["expected_keywords"] )

            results.append({"question":question,
                            "retrieved_chunks":retrieved,
                            "answer":answer,
                            "expected_chunk_id":test_case["expected_chunk_id"],
                            "expected_keywords":test_case["expected_keywords"],
                            "top1_correct":top1_correct,
                            "topk_correct":topk_correct,
                            "answer_correct":answer_correct,
                            "reciprocal_rank":reciprocal_rank,
                            "retrieval_time": retrieval_time,
                            "generation_time": generation_time,
                            "total_time": total_time,})         
                
        top1_accuracy=sum(result["top1_correct"] for result in results)/len(test_cases)
        topk_accuracy=sum(result["topk_correct"] for result in results)/len(test_cases)
        answer_accuracy=sum(result["answer_correct"] for result in results)/len(test_cases)
        avg_reciprocal_rank=sum(result["reciprocal_rank"] for result in results)/len(test_cases)
        avg_retrieval_time = sum(r["retrieval_time"] for r in results) / len(test_cases)
        avg_generation_time = sum(r["generation_time"] for r in results) / len(test_cases)
        avg_total_time = sum(r["total_time"] for r in results) / len(test_cases)
        
        final_result={"detailed_results":results,
                     "metrics":{"top1_accuracy":top1_accuracy,
                                 "topk_accuracy":topk_accuracy,
                                 "answer_accuracy":answer_accuracy,
                                 "mrr":avg_reciprocal_rank,
                                 "avg_retrieval_time": avg_retrieval_time,
                                 "avg_generation_time": avg_generation_time,
                                 "avg_total_time": avg_total_time,
                                 }}
        
        return final_result
           