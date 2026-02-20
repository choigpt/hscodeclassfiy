"""
판결 케이스 정규화 스크립트

Mode A: 규칙 기반 (regex + 패턴 매칭)
Mode B: 규칙 + LLM 하이브리드 (Mode A 후 저확신도 케이스를 LLM으로 보완)

Usage:
    python scripts/normalize_cases.py --mode rule
    python scripts/normalize_cases.py --mode hybrid
    python scripts/normalize_cases.py --mode both
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.attribute_extract import extract_attributes_8axis

DATA_PATH = "data/ruling_cases/all_cases_full_v7.json"
OUTPUT_DIR = "data/ruling_cases"


# ============================================================
# HS Code Parsing
# ============================================================

def parse_hs_code(decision_hs_code: str) -> Dict[str, str]:
    """decision_hs_code에서 HS4/6/10 파싱"""
    result = {"hs4": "", "hs6": "", "hs10": ""}

    if not decision_hs_code:
        return result

    # 한국어 형식: "제5515.12-9000호" or "5515.12-9000"
    # 정규식으로 숫자 추출
    cleaned = re.sub(r'[제호\s]', '', decision_hs_code)

    # 패턴: XXXX.XX-XXXX (full 10-digit)
    m = re.search(r'(\d{4})\.?(\d{2})-?(\d{4})', cleaned)
    if m:
        result["hs4"] = m.group(1)
        result["hs6"] = m.group(1) + m.group(2)
        result["hs10"] = m.group(1) + m.group(2) + m.group(3)
        return result

    # 패턴: XXXX.XX (6-digit)
    m = re.search(r'(\d{4})\.?(\d{2})', cleaned)
    if m:
        result["hs4"] = m.group(1)
        result["hs6"] = m.group(1) + m.group(2)
        return result

    # 패턴: XXXX (4-digit only)
    m = re.search(r'(\d{4})', cleaned)
    if m:
        result["hs4"] = m.group(1)
        return result

    return result


# ============================================================
# Rationale Parsing (GRI / Rejected Codes / Decisive Reasoning)
# ============================================================

def extract_applied_gri(rationale: str) -> List[str]:
    """rationale에서 적용된 GRI 통칙 추출"""
    gri_list = []

    # 한국어 패턴: "통칙 제1호", "일반통칙 제3호", "해석에 관한 통칙"
    patterns = [
        r'통칙\s*제?\s*(\d)\s*호',
        r'일반통칙\s*제?\s*(\d)',
        r'해석.*통칙\s*제?\s*(\d)',
        r'GRI\s*(\d)',
        r'Rule\s*(\d)',
    ]

    for pattern in patterns:
        for m in re.finditer(pattern, rationale, re.IGNORECASE):
            gri_num = m.group(1)
            gri_id = f"GRI{gri_num}"
            if gri_id not in gri_list:
                gri_list.append(gri_id)

    # 소호 분류 언급 시 GRI6 추가
    if re.search(r'소호|제6호|GRI\s*6', rationale, re.IGNORECASE):
        if "GRI6" not in gri_list:
            gri_list.append("GRI6")

    return gri_list


def extract_rejected_codes(rationale: str) -> List[str]:
    """rationale에서 기각된 코드 추출"""
    rejected = []

    # 패턴: "제XXXX호가 아니라", "XXXX호에서 제외", "XXXX호에 해당하지"
    patterns = [
        r'제?(\d{4})\.?(\d{2})?\s*호?\s*(?:가\s*아니|에서\s*제외|에\s*해당하지|로\s*분류하지)',
        r'제?(\d{4})\.?(\d{2})?\s*호?\s*(?:아닌|제외|불해당)',
    ]

    for pattern in patterns:
        for m in re.finditer(pattern, rationale):
            hs4 = m.group(1)
            hs_sub = m.group(2) or ""
            code = hs4 + hs_sub
            if code not in rejected:
                rejected.append(code)

    return rejected


def extract_decisive_reasoning(rationale: str) -> List[str]:
    """rationale에서 핵심 결론문 추출"""
    decisive = []

    # 핵심 결론 패턴: "...분류함", "...분류됨", "...해당됨"
    sentences = re.split(r'[.\n;]', rationale)
    conclusion_patterns = [
        r'분류[하됨]',
        r'해당[하됨]',
        r'적용[하됨]',
        r'결정[하됨]',
    ]

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            continue
        for pattern in conclusion_patterns:
            if re.search(pattern, sent):
                decisive.append(sent[:300])
                break

    return decisive[:3]  # 최대 3개


def extract_essential_character(rationale: str) -> Optional[Dict]:
    """본질적 특성(Essential Character) 언급 추출"""
    ec_patterns = [
        r'본질적\s*특성',
        r'essential\s*character',
        r'주된\s*(?:성분|재질|기능|특성)',
    ]

    for pattern in ec_patterns:
        if re.search(pattern, rationale, re.IGNORECASE):
            return {"mentioned": True}

    return None


# ============================================================
# Feature Extraction
# ============================================================

def extract_features(product_name: str, product_description: str) -> Dict:
    """상품에서 구조화된 features 추출"""
    full_text = f"{product_name} {product_description}".strip()
    attrs = extract_attributes_8axis(full_text)

    materials = []
    for val in attrs.material.values:
        materials.append({"name": val, "ratio": None})

    return {
        "primary_function": attrs.function_use.values[0] if attrs.function_use.values else "",
        "materials": materials,
        "structure": attrs.physical_form.values[0] if attrs.physical_form.values else "",
        "is_set": attrs.is_set(),
        "is_electrical": "electrical" in attrs.function_use.values,
    }


# ============================================================
# Mode A: Rule-based normalization
# ============================================================

def normalize_case_rule(case: Dict, idx: int) -> Dict:
    """단일 케이스 규칙 기반 정규화"""
    ref = case.get("reference_number", f"case_{idx}")
    hs_codes = parse_hs_code(case.get("decision_hs_code", ""))

    # hs_heading fallback
    if not hs_codes["hs4"] and case.get("hs_heading"):
        hs_codes["hs4"] = str(case["hs_heading"])[:4]

    rationale = case.get("rationale", "")
    product_name = case.get("product_name", "")
    product_desc = case.get("product_description", "")

    applied_gri = extract_applied_gri(rationale)
    rejected_codes = extract_rejected_codes(rationale)
    decisive_reasoning = extract_decisive_reasoning(rationale)
    ec = extract_essential_character(rationale)
    features = extract_features(product_name, product_desc)

    # confidence 계산 (regex 매칭 품질)
    confidence = 0.0
    if hs_codes["hs10"]:
        confidence += 0.4
    elif hs_codes["hs6"]:
        confidence += 0.3
    elif hs_codes["hs4"]:
        confidence += 0.2
    if applied_gri:
        confidence += 0.2
    if decisive_reasoning:
        confidence += 0.2
    if features["materials"]:
        confidence += 0.1
    if rejected_codes:
        confidence += 0.1

    return {
        "case_id": f"KR_{ref}",
        "jurisdiction": "KR",
        "hs_version": "2022",
        "final_code_4": hs_codes["hs4"],
        "final_code_6": hs_codes["hs6"],
        "final_code_10": hs_codes["hs10"],
        "rejected_codes_6": rejected_codes,
        "applied_gri": applied_gri,
        "essential_character": ec,
        "features": features,
        "decisive_reasoning": decisive_reasoning,
        "risk_tags": [],
        "_confidence": round(confidence, 2),
        "_source": "rule",
    }


def run_mode_a(cases: List[Dict]) -> List[Dict]:
    """Mode A: 규칙 기반 정규화"""
    print("[Mode A] 규칙 기반 정규화 시작...")
    results = []
    for idx, case in enumerate(cases):
        norm = normalize_case_rule(case, idx)
        results.append(norm)

    print(f"[Mode A] {len(results)} 케이스 정규화 완료")
    return results


# ============================================================
# Mode B: Rule + LLM hybrid normalization
# ============================================================

def call_llm_normalize(case: Dict, rule_result: Dict) -> Optional[Dict]:
    """LLM을 사용한 보완 정규화 (Ollama Qwen2.5)"""
    try:
        import requests
    except ImportError:
        return None

    rationale = case.get("rationale", "")
    product_name = case.get("product_name", "")

    prompt = f"""다음 관세 품목분류 판결의 rationale을 분석하여 JSON으로 구조화하세요.

품명: {product_name}
판결문:
{rationale[:2000]}

다음 JSON 형식으로 응답하세요 (JSON만 출력, 설명 없이):
{{
  "applied_gri": ["GRI1", "GRI6"],
  "rejected_codes": [],
  "decisive_reasoning": ["핵심 결론 1문장"],
  "essential_character": null
}}"""

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 500},
            },
            timeout=30,
        )
        if resp.status_code == 200:
            text = resp.json().get("response", "")
            # JSON 추출
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
    except Exception:
        pass

    return None


def run_mode_b(cases: List[Dict], rule_results: List[Dict]) -> List[Dict]:
    """Mode B: 규칙 + LLM 하이브리드"""
    print("[Mode B] 하이브리드 정규화 시작...")

    hybrid_results = []
    llm_used = 0
    llm_failed = 0

    for idx, (case, rule_result) in enumerate(zip(cases, rule_results)):
        result = dict(rule_result)  # copy

        # 확신도 낮은 케이스에 LLM 보완
        if rule_result["_confidence"] < 0.5:
            llm_output = call_llm_normalize(case, rule_result)
            if llm_output:
                llm_used += 1
                # LLM 결과로 보완 (없는 필드만)
                if not result["applied_gri"] and llm_output.get("applied_gri"):
                    result["applied_gri"] = llm_output["applied_gri"]
                if not result["rejected_codes_6"] and llm_output.get("rejected_codes"):
                    result["rejected_codes_6"] = llm_output["rejected_codes"]
                if not result["decisive_reasoning"] and llm_output.get("decisive_reasoning"):
                    result["decisive_reasoning"] = llm_output["decisive_reasoning"]
                if not result["essential_character"] and llm_output.get("essential_character"):
                    result["essential_character"] = llm_output["essential_character"]
                result["_source"] = "hybrid"
                result["_confidence"] = min(1.0, result["_confidence"] + 0.2)
            else:
                llm_failed += 1

        hybrid_results.append(result)

        if (idx + 1) % 100 == 0:
            print(f"  진행: {idx+1}/{len(cases)}")

    print(f"[Mode B] {len(hybrid_results)} 케이스 완료 (LLM 사용: {llm_used}, 실패: {llm_failed})")
    return hybrid_results


# ============================================================
# Comparison & Output
# ============================================================

def compute_comparison(rule_results: List[Dict], hybrid_results: List[Dict]) -> Dict:
    """두 모드 비교 통계"""
    stats = {
        "total_cases": len(rule_results),
        "rule_mode": {
            "avg_confidence": 0.0,
            "with_gri": 0,
            "with_hs10": 0,
            "with_hs6": 0,
            "with_rejected": 0,
            "with_decisive": 0,
            "with_ec": 0,
        },
        "hybrid_mode": {
            "avg_confidence": 0.0,
            "with_gri": 0,
            "with_hs10": 0,
            "with_hs6": 0,
            "with_rejected": 0,
            "with_decisive": 0,
            "with_ec": 0,
        },
    }

    for mode_key, results in [("rule_mode", rule_results), ("hybrid_mode", hybrid_results)]:
        confidences = []
        for r in results:
            confidences.append(r.get("_confidence", 0))
            if r.get("applied_gri"):
                stats[mode_key]["with_gri"] += 1
            if r.get("final_code_10"):
                stats[mode_key]["with_hs10"] += 1
            if r.get("final_code_6"):
                stats[mode_key]["with_hs6"] += 1
            if r.get("rejected_codes_6"):
                stats[mode_key]["with_rejected"] += 1
            if r.get("decisive_reasoning"):
                stats[mode_key]["with_decisive"] += 1
            if r.get("essential_character"):
                stats[mode_key]["with_ec"] += 1

        stats[mode_key]["avg_confidence"] = round(sum(confidences) / len(confidences), 3) if confidences else 0

    return stats


def save_results(results: List[Dict], output_path: str):
    """결과를 JSONL로 저장"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  저장: {out} ({len(results)} 건)")


def main():
    parser = argparse.ArgumentParser(description="판결 케이스 정규화")
    parser.add_argument("--mode", choices=["rule", "hybrid", "both"], default="both")
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()

    # 데이터 로드
    print(f"데이터 로드: {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        cases = json.load(f)
    print(f"  총 {len(cases)} 케이스")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mode A: Rule-based
    rule_results = run_mode_a(cases)
    save_results(rule_results, str(output_dir / "normalized_cases_rule.jsonl"))

    if args.mode in ["hybrid", "both"]:
        # Mode B: Hybrid
        hybrid_results = run_mode_b(cases, rule_results)
        save_results(hybrid_results, str(output_dir / "normalized_cases_hybrid.jsonl"))

        # 비교 통계
        comparison = compute_comparison(rule_results, hybrid_results)
        comp_path = output_dir / "normalization_comparison.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"\n비교 통계 저장: {comp_path}")
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
    else:
        # rule-only mode: 비교 통계 생성 (self-comparison)
        comparison = compute_comparison(rule_results, rule_results)
        comp_path = output_dir / "normalization_comparison.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"\n통계 저장: {comp_path}")
        print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
