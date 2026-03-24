#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
import uuid
from typing import Dict, Set, Any, Optional

from openai import OpenAI

SYSTEM_PROMPT = (
    "Ти даєш короткі відповіді українською мовою на питання про політику та історію України ХХ–ХХІ століття. "
    "Відповідь має бути 1–4 речення. "
    "Не перераховуй довгі списки статей, прізвищ чи пунктів. "
    "Якщо тема надто широка, дай стислий узагальнений висновок і за потреби 1–3 приклади. "
    "Поверни лише JSON за схемою."
)

MAX_ANSWER_CHARS = 320

def qhash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]

def load_answered_hashes(out_path: str) -> Set[str]:
    answered: Set[str] = set()
    if not os.path.exists(out_path):
        return answered
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                h = obj.get("q_hash")
                if isinstance(h, str) and h:
                    answered.add(h)
            except Exception:
                continue
    return answered


def build_user_prompt(question: str, answer_char_limit: int, retry_hint: Optional[str] = None) -> str:
    prompt = (
        f'Питання: "{question}"\n'
        f"Дай коротку відповідь українською мовою на 1–4 речення, до {answer_char_limit} символів. "
        "Не роби довгих переліків. Якщо повний перелік буде задовгим, дай стислий підсумок замість нього."
    )
    if retry_hint:
        prompt += f"\nУточнення: {retry_hint}"
    return prompt


def count_sentences(text: str) -> int:
    return sum(text.count(mark) for mark in ".!?")


def validate_answer(answer: str, answer_char_limit: int) -> Optional[str]:
    if not answer:
        return "Порожня відповідь"
    if len(answer) > answer_char_limit:
        return f"Відповідь задовга: {len(answer)} символів"
    sentence_count = count_sentences(answer)
    if sentence_count == 0:
        return "Відповідь має містити хоча б одне завершене речення"
    if sentence_count > 4:
        return f"Забагато речень: {sentence_count}"
    return None


def append_failure(fail_path: str, qobj: Dict[str, Any], error: str, raw_output: str) -> None:
    failure_rec = {
        "q_id": qobj.get("id"),
        "question": qobj.get("question"),
        "error": error,
        "raw_output": raw_output,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(fail_path, "a", encoding="utf-8") as fail_file:
        fail_file.write(json.dumps(failure_rec, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions.jsonl", help="Input JSONL with questions")
    ap.add_argument("--out", default="qa.jsonl", help="Output JSONL with question+answer")
    ap.add_argument("--fail-log", default="failed_answers.jsonl", help="Output JSONL with failed generations")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max", type=int, default=0, help="Max new answers this run (0 = no limit)")
    ap.add_argument("--max-output-tokens", type=int, default=220)
    ap.add_argument("--answer-char-limit", type=int, default=MAX_ANSWER_CHARS)
    ap.add_argument("--retries", type=int, default=3, help="Attempts per question before logging failure")
    ap.add_argument("--start-qid", default="", help="Resume from this source question id, inclusive")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY env var")

    client = OpenAI()

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }

    answered_hashes = load_answered_hashes(args.out)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    new_count = 0
    seen_start_qid = not args.start_qid

    with open(args.questions, "r", encoding="utf-8") as fin, open(args.out, "a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            qobj = json.loads(line)
            question = (qobj.get("question") or "").strip()
            if not question:
                continue

            if not seen_start_qid:
                if qobj.get("id") == args.start_qid:
                    seen_start_qid = True
                else:
                    continue

            h = qhash(question)
            if h in answered_hashes:
                continue

            answer = ""
            raw = ""
            failure_reason: Optional[str] = None

            for attempt in range(1, args.retries + 1):
                retry_hint = None
                if attempt > 1:
                    retry_hint = (
                        f"Попередня спроба не пройшла перевірку: {failure_reason}. "
                        "Стисни відповідь ще сильніше і не намагайся давати повний перелік."
                    )

                user_prompt = build_user_prompt(question, args.answer_char_limit, retry_hint)

                resp = client.responses.create(
                    model=args.model,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "short_answer",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                )

                raw = getattr(resp, "output_text", "") or ""
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    failure_reason = f"Invalid JSON: {exc.msg}"
                    with open("last_bad_answer_response.txt", "w", encoding="utf-8") as dbg:
                        dbg.write(raw)
                    continue

                answer = (data.get("answer") or "").strip()
                failure_reason = validate_answer(answer, args.answer_char_limit)
                if failure_reason is None:
                    break

                with open("last_bad_answer_response.txt", "w", encoding="utf-8") as dbg:
                    dbg.write(raw)

            if failure_reason is not None:
                append_failure(args.fail_log, qobj, failure_reason, raw)
                print(f"[WARN] skipped q_id={qobj.get('id')} after {args.retries} attempts: {failure_reason}")
                continue

            out_rec: Dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "q_id": qobj.get("id"),
                "q_hash": h,
                "question": question,
                "answer": answer,
                "meta": {
                    "topic": (qobj.get("meta") or {}).get("topic"),
                    "model": args.model,
                    "temperature": args.temperature,
                    "created_at": now,
                },
            }

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            answered_hashes.add(h)
            new_count += 1

            if new_count % 20 == 0:
                print(f"[OK] generated {new_count} answers...")

            if args.max and new_count >= args.max:
                break

    print(f"Done. New answers written: {new_count}. Output: {args.out}")

if __name__ == "__main__":
    main()