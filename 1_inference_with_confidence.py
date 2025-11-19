import ray
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from typing import List, Dict
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

@dataclass
class LLMConfig():
    model: str = "/data/pretrained_models/Qwen3/Qwen3-8B"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    tensor_parallel_size: int = 1  
    max_logprobs: int = 25
    dtype: str = "float16"

@dataclass
class SamplingConfig():
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    logprobs: int = 25
    prompt_logprobs: int = 0
    n: int = 8

@dataclass
class RayConfig():
    num_actors: int = 2
    num_gpus: int = 2

@dataclass
class DataConfig():
    input_file: str = "/data/zez/Boundary/temp/guardreasoner-prepare.jsonl"
    output_file: str = "/data/zez/Boundary/temp/guardreasoner-metric.jsonl"
    temp_path: str = "/data/zez/Boundary/temp/simple_instruction"
    num_queries: int = None
    random_sample: bool = True
    batch_processing_size: int = 10000


def load_dataset(file_path: str, num_queries: int = None, random_sample: bool = False) -> List[Dict]:
    import random
    random.seed(42)
    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            if num_queries is not None:
                if random_sample:
                    df = df.sample(n=num_queries, random_state=42)
                else:
                    df = df.head(num_queries)
            return df.to_dict('records')

        elif file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            
            if num_queries is not None:
                if random_sample:
                    data = random.sample(data, min(num_queries, len(data)))
                else:
                    data = data[:num_queries]
            return data

        else:
            raise ValueError(f"Unsupport File Format: {file_path}")

    except Exception as e:
        raise ValueError(f"Unable to load file: {file_path}, Exception: {str(e)}")

def save_data(data: list, file_path: str):
    """保存数据"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif path.suffix == '.jsonl':
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    elif path.suffix == '.parquet':
        pd.DataFrame(data).to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


# 使用Actor模式预加载模型
@ray.remote(num_gpus=LLMConfig.tensor_parallel_size, num_cpus=4)
class LogprobInferenceActor:
    def __init__(self):
        
        llm_config = LLMConfig()
        sampling_config = SamplingConfig()
        
        self.llm = LLM(
            model=llm_config.model,
            tensor_parallel_size=llm_config.tensor_parallel_size,
            gpu_memory_utilization=llm_config.gpu_memory_utilization,
            max_model_len=llm_config.max_model_len,
            dtype=llm_config.dtype,
            trust_remote_code=True,
            enforce_eager=True,
            max_logprobs=llm_config.max_logprobs,
        )
        
        self.sampling_params = SamplingParams(
            temperature=sampling_config.temperature,
            top_p=sampling_config.top_p,
            max_tokens=sampling_config.max_tokens,
            logprobs=sampling_config.logprobs,
            n=sampling_config.n,
        )
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """批量生成文本"""

        outputs = self.llm.generate(prompts, self.sampling_params)
        
        results = []
        for output in outputs:
            prompt_candidates = []
            for resp in output.outputs:
                candidate_dict = {"text": resp.text}
                if hasattr(resp, "logprobs") and resp.logprobs is not None:
                    token_confidences = []
                    for pos_logprobs in resp.logprobs:
                        if pos_logprobs:
                            token_confidence = - sum([pos_logprob.logprob for pos_logprob in pos_logprobs.values()]) / len(pos_logprobs)
                        else:
                            token_confidence = None  # 如果该位置没有 logprobs
                        token_confidences.append(token_confidence)
                    candidate_dict["token_confidence"] = token_confidences
                else:
                    candidate_dict["token_confidence"] = None
                prompt_candidates.append(candidate_dict)
            results.append(prompt_candidates)

        return results


def distribute_prompts(prompts: List[str], num_models: int) -> List[List[str]]:
    """将prompts平均分配给模型"""
    distributed = [[] for _ in range(num_models)]
    for i, prompt in enumerate(prompts):
        model_index = i % num_models
        distributed[model_index].append(prompt)
    return distributed


def main():
    if LLMConfig.tensor_parallel_size * RayConfig.num_actors != RayConfig.num_gpus:
        raise ValueError(
            f"Tensor parallel size * num_actors must equal num_gpus, "
            f"got {LLMConfig.tensor_parallel_size} * {RayConfig.num_actors} != {RayConfig.num_gpus}"
        )
    
    # 初始化Ray
    ray.init(num_gpus=RayConfig.num_gpus)
    
    tokenizer = AutoTokenizer.from_pretrained(LLMConfig.model, trust_remote_code=True)
    messages = load_dataset(DataConfig.input_file)

    # 创建模型实例
    tqdm.write(f"🧠 Initializing {RayConfig.num_actors} actors with model {LLMConfig.model} ...")
    models = []
    for i in range(RayConfig.num_actors):
        model = LogprobInferenceActor.options(name=f"logprob_actor_{i+1}").remote()
        models.append(model)

    # === 准备输出目录 ===
    temp_dir = Path(DataConfig.output_file).parent / Path(DataConfig.temp_path)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # === 检查已有的 batch 进度（resume） ===
    existing_parts = sorted(temp_dir.glob("part_*.jsonl"))
    if existing_parts:
        completed_batches = [int(p.stem.split("_")[1]) for p in existing_parts if p.stem.split("_")[1].isdigit()]
        resume_from = max(completed_batches) + 1 if completed_batches else 0
        tqdm.write(f"🔄 Resuming from batch {resume_from} (found {len(existing_parts)} completed parts)")
    else:
        resume_from = 0
        tqdm.write("🚀 Starting from scratch (no existing partial results found)")

    # === 分批处理 ===
    batch_size = DataConfig.batch_processing_size if DataConfig.batch_processing_size is not None else len(messages)
    num_batches = (len(messages) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(resume_from, num_batches),
                        desc="🚀 Processing batches",
                        unit="batch",
                        position=0,       # 外层进度条位置
                        leave=True,
                        dynamic_ncols=True):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(messages))
        batch_messages = messages[start:end]

        tqdm.write(f"📦 Batch {batch_idx + 1}/{num_batches} "
                   f"({start} ~ {end - 1}, total {len(batch_messages)} prompts)")

        prompts = []
        batch_results = []

        for item in tqdm(batch_messages,
                        desc="Tokenize Prompt",
                        position=1,      # 内层进度条位置
                        leave=False,
                        dynamic_ncols=True):
            prompt = tokenizer.apply_chat_template(
                item["conversations"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )

            batch_results.append(item)
            prompts.append(prompt)

        # 将prompts平均分配给模型
        distributed_prompts = distribute_prompts(prompts, RayConfig.num_actors)
        
        # 为每个模型创建批量推理任务
        tasks = []
        for i, (model, batch_prompts) in enumerate(zip(models, distributed_prompts)):
            print(f"Model {i+1} will process {len(batch_prompts)} prompts")
            if batch_prompts:  # 只有当有prompts时才创建任务
                task = model.generate_batch.remote(batch_prompts)
                tasks.append(task)
            else:
                tasks.append(None)  # 没有prompts的模型
        
        # 批量获取所有结果
        all_results = []
        for i, task in enumerate(tasks):
            if task is not None:
                part_results = ray.get(task)
            all_results.extend(part_results)
        
        # 将结果添加到数据中（保持原始顺序）
        prompt_to_result = {}
        result_index = 0
        
        # 重新分配结果到对应的prompt
        for i, batch_prompts in enumerate(distributed_prompts):
            for j in range(len(batch_prompts)):
                if result_index < len(all_results):
                    # 找到这个prompt在原始prompts中的位置
                    original_index = i + j * RayConfig.num_actors
                    if original_index < len(prompts):
                        prompt_to_result[original_index] = all_results[result_index]
                        result_index += 1
        
        # 按照原始顺序整理结果
        final_results = []
        for i in range(len(prompts)):
            final_results.append(prompt_to_result.get(i, "Error: Result not found"))
        
        # 将结果添加到数据中
        for item, result in zip(batch_results, final_results):
            item['candidates'] = result

        # 保存临时文件
        part_file = temp_dir / f"part_{batch_idx:03d}.jsonl"
        tqdm.write(f"💾 Saving partial results to {part_file}")
        save_data(batch_results, part_file)

 
    # === 合并所有临时文件 ===
    output_file = Path(DataConfig.output_file)
    print(f"\n🔗 Merging all parts into {output_file}")
    with open(output_file, 'w', encoding='utf-8') as fout:
        for part_file in sorted(temp_dir.glob("part_*.jsonl")):
            with open(part_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)

    print("✅ All batches processed and merged successfully!")

    
    ray.shutdown()


if __name__ == "__main__":
    main()
