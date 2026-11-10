import os
import json
import time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# =========================
# 1. Paths
# =========================
base_dir = Path(__file__).resolve().parent
prompt_dir = base_dir / "prompt"
output_dir = base_dir / "json"
env_file = base_dir / ".env"

output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 2. Load .env
# =========================
load_dotenv(dotenv_path=env_file)
api_key = os.environ.get("DASHSCOPE_API_KEY")

# =========================
# 3. Settings
# =========================
MODEL_NAME = "qwen3.7-flash"
WORKERS = 5                 # Number of concurrent requests
ENABLE_THINKING = False     # Disable thinking mode
TEMPERATURE = 0.7           # Qwen3.7 default without thinking
SKIP_EXISTING = True        # Skip existing JSON files
MAX_RETRIES = 3             # Maximum retry attempts
RETRY_DELAY = 2             # Seconds between retries

# System prompt
SYSTEM_PROMPT = (
    "You are an AI assistant. Your task is to answer the 'New Survey Question' "
    "as if you are the person described in the 'Persona Profile' (which consists "
    "of their past survey responses). Follow all instructions provided for the "
    "new question carefully regarding the format of your answer."
)

# =========================
# 4. Helper functions
# =========================
def build_client():
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is missing. Set it in your environment or in a local .env file."
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

def process_one_file(file_path: Path):
    """
    Read a prompt file, call the model, and save the JSON response.
    """
    pid = file_path.stem.removeprefix("pid_").removesuffix("_prompt")

    output_file = output_dir / f"{pid}.json"

    if SKIP_EXISTING and output_file.exists():
        return {
            "status": "skipped",
            "pid": pid,
            "file": file_path.name,
            "message": f"{output_file.name} already exists; skipped"
        }

    prompt_text = file_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        return {
            "status": "empty",
            "pid": pid,
            "file": file_path.name,
            "message": "Prompt file is empty"
        }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = build_client()

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=TEMPERATURE,
                extra_body={"enable_thinking": ENABLE_THINKING},
                stream=False
            )

            result = {
                "pid": pid,
                "source_file": file_path.name,
                "model": MODEL_NAME,
                "enable_thinking": ENABLE_THINKING,
                "temperature": TEMPERATURE,
                "system_prompt": SYSTEM_PROMPT,
                "content": response.choices[0].message.content
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return {
                "status": "success",
                "pid": pid,
                "file": file_path.name,
                "message": f"Saved to {output_file.name}"
            }

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                error_file = output_dir / f"{pid}_error.json"
                error_info = {
                    "pid": pid,
                    "source_file": file_path.name,
                    "error": last_error
                }
                with open(error_file, "w", encoding="utf-8") as f:
                    json.dump(error_info, f, ensure_ascii=False, indent=2)

    return {
        "status": "error",
        "pid": pid,
        "file": file_path.name,
        "message": last_error
    }

# =========================
# 5. Main program
# =========================
def main():
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is missing. Set it in your environment or in a local .env file."
        )
    prompt_files = sorted(prompt_dir.glob("pid_*_prompt.txt"))
    if not prompt_files:
        raise FileNotFoundError(f"No pid_*_prompt.txt files found in {prompt_dir}")

    print(f"Prompt files: {len(prompt_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {MODEL_NAME}")
    print(f"Thinking: {ENABLE_THINKING}")
    print(f"Workers: {WORKERS}")
    print(f"Temperature: {TEMPERATURE}")
    print("-" * 60)

    results = []

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_file = {
            executor.submit(process_one_file, file_path): file_path
            for file_path in prompt_files
        }

        with tqdm(total=len(future_to_file), desc="Calling Qwen", ncols=100) as pbar:
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)

                status = result["status"]
                pid = result["pid"]

                pbar.set_postfix_str(f"{status} | pid={pid}")
                pbar.update(1)

                print(f"[{status}] pid={pid} | {result['message']}")

    print("\nProcessing complete")
    counts = Counter(result["status"] for result in results)
    print(f"Succeeded: {counts['success']}")
    print(f"Skipped: {counts['skipped']}")
    print(f"Failed: {counts['error']}")

    # Save the run summary
    summary_file = output_dir / "_run_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Run summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
