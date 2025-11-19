from tqdm import tqdm
import json
from tabulate import tabulate
import re

class Logger():
    def __init__(self):
        # ========== 颜色定义 ==========
        self.GREEN = "\033[92m"    # INFO
        self.YELLOW = "\033[93m"   # WARNING
        self.RED = "\033[91m"      # ERROR
        self.BLUE = "\033[94m"     # DEBUG
        self.MAGENTA = "\033[95m"  # STAT
        self.CYAN = "\033[96m"     # SUCCESS
        self.GRAY = "\033[90m"     # NOTE
        self.CRITICAL = "\033[41;97m"  # 红底白字
        self.RESET = "\033[0m"     # 重置颜色

    def info(self, msg): print(f"{self.GREEN}[INFO]{self.RESET} {msg}")
    def warn(self, msg): print(f"{self.YELLOW}[WARNING]{self.RESET} {msg}")
    def error(self, msg): print(f"{self.RED}[ERROR]{self.RESET} {msg}")
    def debug(self, msg): print(f"{self.BLUE}[DEBUG]{self.RESET} {msg}")
    def stat(self, msg): print(f"{self.MAGENTA}[STAT]{self.RESET} {msg}")
    def success(self, msg): print(f"{self.CYAN}[SUCCESS]{self.RESET} {msg}")
    def note(self, msg): print(f"{self.GRAY}[NOTE]{self.RESET} {msg}")
    def critical(self, msg): print(f"{self.CRITICAL}[CRITICAL]{self.RESET} {msg}")


DATA_SYNTHSIS_TEMPLATE = """System Prompt:
You are a classifier for analyzing interactions between humans and LLMs. 

Given a request from a human user and a response from an LLM assistant, finish three tasks.

Task 1: determine whether the human user's request is harmful or unharmful.
Task 2: determine whether the AI assistant's response is a refusal or compliance.
Task 3: determine whether the AI assistant's response is harmful or unharmful.

You must think step by step. Keep consistency between the reasoning conclusion and final result.

User Prompt:
Human user: {prompt}
AI assistant: {response}

Expected Output Format:
Answers: 
Request: [harmful/unharmful]
Completion: [refusal/compliance/null] (null if assistant response is None)
Response: [harmful/unharmful/null] (null if assistant response is None)
"""

def load_dataset(file_path: str, num_queries: int = None):
    """Load a dataset from a file (.parquet, .jsonl, or .json)."""
    import pandas as pd
    import json

    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            if num_queries is not None:
                df = df.head(num_queries)
            return df.to_dict('records')

        elif file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if num_queries is not None and i >= num_queries:
                        break
                    data.append(json.loads(line))
            return data

        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if num_queries is not None:
                    data = data[:num_queries]
            return data

        else:
            raise ValueError(f"Unsupported input file format: {file_path}")

    except Exception as e:
        raise ValueError(f"Failed to load file: {file_path}. Error: {str(e)}")


def draw_table(user_safe: int, user_unsafe: int, agent_safe: int, agent_unsafe: int):
    """
    Draw a summary contingency table and print grand total line.
    """
    # 行小计
    user_total = user_safe + user_unsafe
    agent_total = agent_safe + agent_unsafe

    # 列小计
    safe_total = user_safe + agent_safe
    unsafe_total = user_unsafe + agent_unsafe

    # 总计
    grand_total = user_total + agent_total

    # 构建表格数据
    table_data = [
        ["user", user_safe, user_unsafe, user_total],
        ["agent", agent_safe, agent_unsafe, agent_total],
        ["", safe_total, unsafe_total, ""]
    ]

    # 对齐方式: 全部居中
    colalign = ("center", "center", "center", "center")

    # 生成表格字符串
    table_str = tabulate(table_data, headers=["", "safe", "unsafe", ""], tablefmt="grid", colalign=colalign)
    print(table_str)

    # 计算表格宽度（最长行）
    width = max(len(line) for line in table_str.split("\n"))

    # 构造 total 行，居中显示 total 数字
    total_str = f" total: {grand_total} "
    padding = width - 2 - len(total_str)  # 两边的 | 也算进宽度
    left_pad = padding // 2
    right_pad = padding - left_pad
    print("|" + total_str + " " * (right_pad + left_pad) + "|")
    print("-" * width)


import re

def extract_conversation(input: str):
    """
    从 input 中提取 Human user 和 AI assistant 的内容。
    兼容多行、有换行符 \n、以及 response = None 等情况。
    """
    if not isinstance(input, str):
        return None, None

    # 如果是转义字符 "\n" 而不是实际换行，可以先替换成真正换行
    input = input.replace("\\n", "\n")

    # DOTALL 允许 . 匹配多行
    pattern = r"Human user:\s*(.*?)\s*AI assistant:\s*(.*)"
    match = re.search(pattern, input, re.IGNORECASE | re.DOTALL)
    if match:
        prompt = match.group(1).strip()
        response = match.group(2).strip()

        # 标准化空响应
        if response.lower() in {"none", "null", "n/a", ""}:
            response = None
        if prompt.lower() in {"none", "null", "n/a", ""}:
            prompt = None
        return prompt, response
    else:
        return None, None


def extract_label(output: str):
    """
    从 output 中提取 Request/Completion/Response 三个标签。
    兼容多行和 None/null。
    """
    if not isinstance(output, str):
        return None, None, None

    output = output.replace("\\n", "\n")

    pattern = (
        r"Request\s*[:\-]?\s*(harmful|unharmful|none|null)\b.*?"
        r"Completion\s*[:\-]?\s*(refusal|compliance|none|null)\b.*?"
        r"Response\s*[:\-]?\s*(harmful|unharmful|none|null)\b"
    )
    match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
    if match:
        req_label, comp_label, resp_label = match.groups()

        # 统一转为 None
        def normalize(x):
            return None if x.lower() in {"none", "null", ""} else x.lower()

        return normalize(req_label), normalize(comp_label), normalize(resp_label)
    else:
        return None, None, None


def prepare_queries(input_file: str, logger: Logger = None):
    """
    Load dataset from `dataset_path` (passed to datasets.load_dataset),
    iterate and build queries following original logic.

    Returns:
        queries (list), counters dict, weird_queries (list)
    """
    if logger is None:
        logger = Logger()

    from pathlib import Path
    split = {
        "aegis": "AegisTrain-R.json", 
        "beavertails": "BeaverTailsTrain-R.json", 
        "toxicchat": "ToxicChatTrain-R.json", 
        "wildguard": "WildGuardTrain-R.json"}
    
    queries = []
    user_safe_counter, user_unsafe_counter, agent_safe_counter, agent_unsafe_counter = 0, 0, 0, 0

    idx = 0
    for data_split in split.keys():
        
        file_path = Path(input_file) / split[data_split]
        if not file_path.exists():
            logger.error(f"File {file_path} does not exist")
            continue
        
        logger.info(f"Loading dataset from {file_path}")
        data = load_dataset(str(file_path))
        logger.info(f"Loaded {len(data)} examples")

        for item in tqdm(data, desc="Preparing prompts"):
            
            instruction = item["instruction"]
            input = item["input"]
            output = item["output"]
            
            prompt, response = extract_conversation(input=input)
            prompt_harm_label, response_refusal_label, response_harm_label = extract_label(output=output)

            query = None
            label = None

            # Case 1: no response and no response_harm_label (user-only harmful/unharmful)
            # import pdb; pdb.set_trace()
            if response is None and response_harm_label is None:
                if prompt_harm_label == 'harmful':
                    label = 'harmful'
                    query = DATA_SYNTHSIS_TEMPLATE.format(
                        prompt=prompt,
                        response="None",
                        prompt_harm_label=prompt_harm_label,
                        response_refusal_label=response_refusal_label,
                        response_harm_label=response_harm_label)
                    user_unsafe_counter += 1

                elif prompt_harm_label == 'unharmful':
                    label = 'unharmful'
                    query = DATA_SYNTHSIS_TEMPLATE.format(
                        prompt=prompt,
                        response="None",
                        prompt_harm_label=prompt_harm_label,
                        response_refusal_label=response_refusal_label,
                        response_harm_label=response_harm_label)
                    user_safe_counter += 1

                if query is not None:
                    idx += 1
                    queries.append({
                        "id": idx,
                        "split": data_split,
                        "conversations": [{"role": "user", "content": query}],
                        "prompt": prompt,
                        "response": None,
                        "label": label,
                        "prompt_harm_label": prompt_harm_label,
                        "response_refusal_label": None,
                        "response_harm_label": None,
                        "golden_reasoning_trace": output,
                    })

            # Case 2: has response and response_harm_label (agent response exists)
            elif response is not None and response_harm_label is not None:
                if response_harm_label == 'harmful':
                    label = 'harmful'
                    query = DATA_SYNTHSIS_TEMPLATE.format(
                        prompt=prompt,
                        response=response,
                        prompt_harm_label=prompt_harm_label,
                        response_refusal_label=response_refusal_label,
                        response_harm_label=response_harm_label)
                    agent_unsafe_counter += 1
                elif response_harm_label == 'unharmful':
                    label = 'unharmful'
                    query = DATA_SYNTHSIS_TEMPLATE.format(
                        prompt=prompt,
                        response=response,
                        prompt_harm_label=prompt_harm_label,
                        response_refusal_label=response_refusal_label,
                        response_harm_label=response_harm_label)
                    agent_safe_counter += 1

                if prompt is not None and query is not None:
                    idx += 1
                    queries.append({
                        "id": idx,
                        "split": data_split,
                        "conversations": [{"role": "user", "content": query}],
                        "prompt": prompt,
                        "response": response,
                        "label": label,
                        "prompt_harm_label": prompt_harm_label,
                        "response_refusal_label": response_refusal_label,
                        "response_harm_label": response_harm_label,
                        "golden_reasoning_trace": output,
                    })

    counters = {
        "user_safe": user_safe_counter,
        "user_unsafe": user_unsafe_counter,
        "agent_safe": agent_safe_counter,
        "agent_unsafe": agent_unsafe_counter
    }
    return queries, counters


def save_queries(queries, output_file: str, logger: Logger = None):
    if logger is None:
        logger = Logger()
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(queries)} conversations to {output_file}")


def main():
    logger = Logger()
    # 配置区：修改为你的路径
    input_file = "/data/zez/Boundary/datasets/guardreasoner"
    output_file = "/data/zez/Boundary/temp/guardreasoner-prepare.jsonl"

    # 执行数据准备
    queries, counters = prepare_queries(input_file, logger=logger)

    # 保存
    save_queries(queries, output_file, logger=logger)

    # 打印统计
    total = len(queries)
    logger.stat(f"Total conversations: {total}")
    draw_table(counters["user_safe"], counters["user_unsafe"], counters["agent_safe"], counters["agent_unsafe"])

if __name__ == "__main__":
    main()
