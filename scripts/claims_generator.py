#!/usr/bin/env python3
"""
claims_generator.py

Reads qa.jsonl (question+answer pairs) and extracts atomic, verifiable claims
from each answer using an LLM.

Output format per record:
{
  "id": "uk_0001",
  "input": "<answer text>",
  "output": {
    "claims": ["<claim1>", "<claim2>", ...]
  },
  "meta": {
    "q_id": "...",
    "q_hash": "...",
    "question": "...",
    "topic": "...",
    "model": "...",
    "temperature": 0.1,
    "created_at": "..."
  }
}
"""

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Set

from openai import OpenAI

SYSTEM_PROMPT = (
    "Ти — аналітик тверджень (claim extraction system). "
    "Твоє завдання — виділити з тексту ВСІ перевірювані фактичні або атрибутовані твердження "
    "з максимальним покриттям і перетворити їх на атомарні, самодостатні claims.\n"
    "Правила:\n"
    "1. Кожен claim має містити рівно ОДИН факт або ОДНУ атрибутовану заяву.\n"
    "2. Кожен claim має бути самодостатнім і зрозумілим без зовнішнього контексту.\n"
    "3. Розгортай займенники та нечіткі посилання: замість «він», «вона», «вони», «це», «там», "
    "«тоді» використовуй повне ім'я, назву організації, подію, місце або дату, якщо це можна "
    "однозначно відновити з тексту.\n"
    "4. Включай усі перевірювані твердження, зокрема:\n"
    "   - дати, часові прив’язки, періоди;\n"
    "   - події та дії осіб, організацій, держав, інституцій;\n"
    "   - рішення, закони, укази, угоди, документи;\n"
    "   - числові дані, відсотки, кількості, результати;\n"
    "   - причинно-наслідкові твердження, якщо вони явно сформульовані в тексті;\n"
    "   - атрибутовані твердження: «X заявив, що...», «Y повідомило, що...», "
    "«за даними Z...», «A вважає, що...», «B стверджує, що...».\n"
    "5. Якщо твердження подано як думку, позицію, оцінку або заяву певного суб’єкта, "
    "ОБОВ’ЯЗКОВО зберігай атрибуцію. Не перетворюй заяву на безособовий факт.\n"
    "6. Якщо одне речення містить кілька окремих фактів або кілька окремих заяв, "
    "розбий його на кілька claims: один claim = один факт/одна заява.\n"
    "7. Якщо речення містить і факт, і наслідок, і причину — за можливості розбий їх "
    "на окремі claims.\n"
    "8. Не об’єднуй різні події, різні дати, різні дії або різних суб’єктів в один claim.\n"
    "9. Виключай лише такі фрагменти:\n"
    "   - вступні слова й пусті конструкції без фактичного змісту: «загалом», "
    "«важливо зазначити», «варто пам’ятати», «як відомо»;\n"
    "   - метакоментарі: «я думаю», «мені здається», «можна сказати», якщо вони не містять "
    "атрибутованої перевірюваної інформації;\n"
    "   - риторичні запитання;\n"
    "   - абстрактні узагальнення без конкретного суб’єкта, дії або події.\n"
    "10. Якщо фрагмент містить політичну, історичну або аналітичну інтерпретацію, але вона "
    "прив’язана до конкретного суб’єкта («аналітики вважають», «автор стверджує», "
    "«уряд пояснює це тим, що...»), включай її як атрибутований claim.\n"
    "11. Зберігай мову вхідного тексту. Не перекладай claims.\n"
    "12. Не додавай нових фактів, не виправляй історичну чи політичну точність тексту, "
    "не узагальнюй і не скорочуй зміст настільки, щоб втратити перевірювані деталі.\n"
    "13. Якщо у тексті немає жодного придатного claim, поверни порожній список.\n"
    "14. Поверни лише JSON за схемою:\n"
    "{\n"
    '  "claims": [\n'
    '    "твердження 1",\n'
    '    "твердження 2"\n'
    "  ]\n"
    "}\n"
    "15. Не додавай жодного тексту поза JSON."
)

MAX_CLAIMS = 25
MAX_CLAIM_CHARS = 350


def ahash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def load_processed_hashes(out_path: str) -> Set[str]:
    processed: Set[str] = set()
    if not os.path.exists(out_path):
        return processed
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                h = (obj.get("meta") or {}).get("q_hash")
                if isinstance(h, str) and h:
                    processed.add(h)
            except Exception:
                continue
    return processed


def load_start_counter(out_path: str) -> int:
    """Return 1 + index of last written record (for uk_XXXX sequential IDs)."""
    count = 0
    if not os.path.exists(out_path):
        return 1
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count + 1


def validate_claims(claims: List[str]) -> Optional[str]:
    if not isinstance(claims, list):
        return "claims не є списком"
    if len(claims) > MAX_CLAIMS:
        return f"Забагато тверджень: {len(claims)}"
    for i, c in enumerate(claims):
        if not isinstance(c, str) or not c.strip():
            return f"Твердження {i} порожнє або не рядок"
        if len(c) > MAX_CLAIM_CHARS:
            return f"Твердження {i} задовге: {len(c)} символів"
    return None


def append_failure(
    fail_path: str, qa_obj: Dict[str, Any], error: str, raw: str
) -> None:
    rec = {
        "q_id": qa_obj.get("q_id"),
        "question": qa_obj.get("question"),
        "answer": qa_obj.get("answer"),
        "error": error,
        "raw_output": raw,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(fail_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_user_prompt(answer: str, retry_hint: Optional[str] = None) -> str:
    prompt = (
        f'Текст: "{answer}"\n\n'
        "Виділи з цього тексту всі перевірювані фактичні твердження відповідно до правил."
    )
    if retry_hint:
        prompt += f"\nУточнення для повторної спроби: {retry_hint}"
    return prompt


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract atomic claims from qa.jsonl answers"
    )
    ap.add_argument("--qa", default="qa.jsonl", help="Source QA JSONL file")
    ap.add_argument("--out", default="claims.jsonl", help="Output claims JSONL file")
    ap.add_argument(
        "--fail-log",
        default="failed_claims.jsonl",
        help="Log for failed records",
    )
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument(
        "--max-output-tokens",
        type=int,
        default=900,
        help="Max tokens per LLM response",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Attempts per record before logging failure and moving on",
    )
    ap.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max new records this run (0 = process all)",
    )
    ap.add_argument(
        "--start-qid",
        default="",
        help="Resume from this source q_id (inclusive)",
    )
    ap.add_argument(
        "--id-prefix",
        default="uk_",
        help="Prefix for sequential output IDs (default: uk_)",
    )
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY env var")

    client = OpenAI()

    schema = {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    }

    processed_hashes = load_processed_hashes(args.out)
    id_counter = load_start_counter(args.out)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    new_count = 0
    seen_start_qid = not args.start_qid

    with open(args.qa, "r", encoding="utf-8") as fin, open(
        args.out, "a", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            qa_obj = json.loads(line)
            q_id = qa_obj.get("q_id") or ""
            answer = (qa_obj.get("answer") or "").strip()
            question = (qa_obj.get("question") or "").strip()

            if not answer:
                continue

            if not seen_start_qid:
                if q_id == args.start_qid:
                    seen_start_qid = True
                else:
                    continue

            h = ahash(answer)
            if h in processed_hashes:
                continue

            claims: List[str] = []
            raw = ""
            failure_reason: Optional[str] = None

            for attempt in range(1, args.retries + 1):
                retry_hint = None
                if attempt > 1:
                    retry_hint = (
                        f"Попередня спроба не пройшла перевірку: {failure_reason}. "
                        "Будь ласка, скороти твердження та переконайся, що кожне — "
                        "окремий перевірюваний факт до 250 символів."
                    )

                user_prompt = build_user_prompt(answer, retry_hint)

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
                            "name": "claims_output",
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
                    with open("last_bad_claims_response.txt", "w", encoding="utf-8") as dbg:
                        dbg.write(raw)
                    continue

                claims = [c.strip() for c in (data.get("claims") or []) if c.strip()]
                failure_reason = validate_claims(claims)
                if failure_reason is None:
                    break

                with open("last_bad_claims_response.txt", "w", encoding="utf-8") as dbg:
                    dbg.write(raw)

            if failure_reason is not None:
                append_failure(args.fail_log, qa_obj, failure_reason, raw)
                print(
                    f"[WARN] skipped q_id={q_id} after {args.retries} attempts: {failure_reason}"
                )
                continue

            # Allow records with zero claims (e.g. answer contained no verifiable facts)
            # but still write them so we can track coverage
            record_id = f"{args.id_prefix}{id_counter:04d}"
            out_rec: Dict[str, Any] = {
                "id": record_id,
                "input": answer,
                "output": {"claims": claims},
                "meta": {
                    "q_id": q_id,
                    "q_hash": h,
                    "question": question,
                    "topic": (qa_obj.get("meta") or {}).get("topic"),
                    "model": args.model,
                    "temperature": args.temperature,
                    "created_at": now,
                },
            }

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            processed_hashes.add(h)
            id_counter += 1
            new_count += 1

            if new_count % 20 == 0:
                print(f"[OK] processed {new_count} records...")

            if args.max and new_count >= args.max:
                break

    print(f"Done. New claims records written: {new_count}. Output: {args.out}")


if __name__ == "__main__":
    main()
