#!/usr/bin/env python3
import argparse
import json
import os
import time
import uuid
from openai import OpenAI

SYSTEM_PROMPT = (
    "Ти генеруєш україномовні питання для датасету про внутрішню та зовнішню політику України у ХХ–ХХІ століттях. "
    "Питання мають бути придатні для відповіді LLM у 5–10 реченнях. "
    "Уникай дублювання змісту. Не пиши відповіді. Не додавай пояснень. "
    "Поверни лише JSON за заданою схемою."
)

def extract_first_json_object(raw: str) -> str:
    """
    Extracts the first complete JSON object from a string by brace matching,
    correctly handling strings and escapes.
    """
    start = raw.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output.")

    in_str = False
    esc = False
    depth = 0

    for i in range(start, len(raw)):
        ch = raw[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start:i+1]

    raise ValueError("JSON appears truncated (no matching closing '}' found).")

def load_topics(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="data/topics_us.txt", help="1 topic per line")
    ap.add_argument("--out", default="questions.jsonl")
    ap.add_argument("--per-topic", type=int, default=20)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY env var")

    client = OpenAI()
    topics = load_topics(args.topics)

    schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": args.per_topic,
                "maxItems": args.per_topic,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "question_type": {"type": "string"},
                        "time_scope": {"type": "string"},
                    },
                    "required": ["question", "question_type", "time_scope"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    }

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with open(args.out, "a", encoding="utf-8") as f:
        for topic in topics:
            user_prompt = (
                f'Топік: "{topic}"\n\n'
                f"Згенеруй рівно {args.per_topic} різних питань по цьому топіку.\n"
                "Питання мають покривати різні типи:\n"
                "- факт/дата/визначення\n"
                "- таймлайн подій\n"
                "- позиції сторін/заяви (атрибуція \"хто що заявляв\")\n"
                "- причини та наслідки (без категоричних суджень)\n\n"
                "Мова: лише українська.\n"
            )

            resp = client.responses.create(
                model=args.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=args.temperature,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "topic_questions",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            try:
                print("finish:", resp.output[0].finish_reason)
            except Exception:
                pass

            # --- robust parse ---
            raw = getattr(resp, "output_text", "") or ""

            try:
                json_str = extract_first_json_object(raw)
                data = json.loads(json_str)
            except Exception:
                with open("last_bad_response.txt", "w", encoding="utf-8") as dbg:
                    dbg.write(raw)
                raise

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Save raw output for debugging
                debug_path = "last_bad_response.txt"
                with open(debug_path, "w", encoding="utf-8") as dbg:
                    dbg.write(raw)
                raise RuntimeError(
                    f"Model returned invalid JSON. Raw response saved to {debug_path}.\n"
                    f"First 500 chars:\n{raw[:500]!r}"
                )
            for q in data["questions"]:
                rec = {
                    "id": str(uuid.uuid4()),
                    "question": q["question"].strip(),
                    "meta": {
                        "topic": topic,
                        "question_type": q["question_type"].strip(),
                        "time_scope": q["time_scope"].strip(),
                        "model": args.model,
                        "temperature": args.temperature,
                        "created_at": now,
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"[OK] {topic}: wrote {len(data['questions'])} questions")

    print(f"Done. Output: {args.out}")

if __name__ == "__main__":
    main()