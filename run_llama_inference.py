# run_llama_inference.py

import re
import ast
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from config import *

def setup_model():
    """모델 및 토크나이저 초기화"""
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=DEVICE_MAP,
        torch_dtype=TORCH_DTYPE,
        use_auth_token=USE_AUTH_TOKEN
    )
    print("Model loaded successfully!")
    return tokenizer, model

def make_prompt(tokenizer, model, context, question, choices):
    """프롬프트 생성"""
    ch = ast.literal_eval(choices)
    user_msg = (
        f"[제공된 정보]: {context}\n\n"
        f"[질문]: {question}\n\n"
        "선택지:\n"
        f"{ch[0]}\n"
        f"{ch[1]}\n"
        f"{ch[2]}\n\n"
        "지침:\n"
        "한 줄에 '[답변]: X' 형태로 단 하나의 정답만을 작성하십시오.\n\n"
        "[답변]:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user_msg},
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    return prompt_ids.to(model.device)

def predict_answer(tokenizer, model, context, question, choices):
    """답변 예측"""
    input_ids = make_prompt(tokenizer, model, context, question, choices)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = gen_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return pd.Series({
        "raw_input": tokenizer.decode(input_ids[0], skip_special_tokens=True),
        "raw_output": generated_text,
        "answer": generated_text
    })

def extract_answer(text, valid_choices):
    """답변 추출"""
    m = re.search(r"\[?답변\]?\s*[:：]\s*([^\n\r]+)", text)
    if m:
        return m.group(1).strip()
    
    found = []
    for choice in valid_choices:
        pattern = rf"\b{re.escape(choice)}\b"
        if re.search(pattern, text):
            found.append(choice)
    
    if len(set(found)) == 1:
        return found[0]
    return None

def main():
    """메인 실행 함수"""
    # 데이터 로드
    print("Loading data...")
    data = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    print(f"Loaded {len(data)} rows")
    
    # 모델 초기화
    tokenizer, model = setup_model()
    
    # 처리 시작
    print("Starting inference...")
    with open(OUTPUT_TXT, "w", encoding="utf-8") as fout:
        for i in tqdm(range(len(data)), total=len(data), ncols=80, desc="Processing"):
            row = data.loc[i]
            
            # 예측 수행
            result = predict_answer(tokenizer, model, row["context"], row["question"], row["choices"])
            raw_output = result["raw_output"]

            # 유효한 선택지 파싱
            try:
                valid_choices = ast.literal_eval(row["choices"])
                if not isinstance(valid_choices, list):
                    valid_choices = []
            except:
                valid_choices = []

            # 답변 추출 및 검증
            ans = extract_answer(raw_output, valid_choices)
            final_answer = ans if ans in valid_choices else "알 수 없음"

            # 결과 저장
            data.at[i, "raw_input"] = result["raw_input"]
            data.at[i, "raw_output"] = raw_output
            data.at[i, "answer"] = final_answer

            # 파일 출력
            fout.write(f"{row['ID']}\n{raw_output}\n{final_answer}\n")

            # 체크포인트 저장
            if i and i % CHECKPOINT_INTERVAL == 0:
                tqdm.write(f"✅ checkpoint @ {i}/{len(data)}")
                checkpoint_path = f"{CHECKPOINT_PREFIX}{i}.csv"
                data[["ID", "raw_input", "raw_output", "answer"]].to_csv(
                    checkpoint_path, index=False, encoding="utf-8-sig"
                )

    # 최종 결과 저장
    print("Saving final results...")
    submission = data[["ID", "raw_input", "raw_output", "answer"]]
    submission.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Completed! Results saved to {FINAL_CSV}")

if __name__ == "__main__":
    main()