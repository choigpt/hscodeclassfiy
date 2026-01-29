"""
Data Split Module - 벤치마크 데이터셋 분할

Stratified + Optional Temporal Split 지원
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import random


@dataclass
class DataSample:
    """단일 데이터 샘플"""
    id: str
    text: str
    hs4: str
    hs6: str = ""
    hs10: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "hs4": self.hs4,
            "hs6": self.hs6,
            "hs10": self.hs10,
            "meta": self.meta
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataSample":
        return cls(
            id=d.get("id", ""),
            text=d.get("text", ""),
            hs4=d.get("hs4", ""),
            hs6=d.get("hs6", ""),
            hs10=d.get("hs10", ""),
            meta=d.get("meta", {})
        )


@dataclass
class SplitStats:
    """분할 통계"""
    total_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    total_classes: int = 0
    train_classes: int = 0
    val_classes: int = 0
    test_classes: int = 0
    dropped_classes: int = 0
    dropped_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "total_classes": self.total_classes,
            "train_classes": self.train_classes,
            "val_classes": self.val_classes,
            "test_classes": self.test_classes,
            "dropped_classes": self.dropped_classes,
            "dropped_samples": self.dropped_samples,
            "train_ratio": self.train_samples / self.total_samples if self.total_samples > 0 else 0,
            "val_ratio": self.val_samples / self.total_samples if self.total_samples > 0 else 0,
            "test_ratio": self.test_samples / self.total_samples if self.total_samples > 0 else 0,
        }


class DataSplitter:
    """
    벤치마크 데이터 분할기

    - Stratified split: 클래스 비율 유지
    - Temporal split: 시간 기반 분할 (옵션)
    - 최소 샘플 수 필터링
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_samples_per_class: int = 3,
        seed: int = 42
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_samples_per_class = min_samples_per_class
        self.seed = seed

        # 비율 검증
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def _generate_id(self, text: str, hs4: str, index: int) -> str:
        """고유 ID 생성"""
        content = f"{text}_{hs4}_{index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _load_ruling_cases(self, path: str) -> List[DataSample]:
        """결정사례 JSON 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            cases = json.load(f)

        samples = []
        for i, case in enumerate(cases):
            hs_code = case.get('hs_code', '') or ''
            hs_heading = case.get('hs_heading', '') or ''
            product_name = case.get('product_name', '').strip()

            # HS4 결정
            if hs_heading and len(hs_heading) == 4:
                hs4 = hs_heading
            elif hs_code and len(hs_code) >= 4:
                hs4 = hs_code[:4]
            else:
                continue

            # 유효성 검사
            if not product_name or len(product_name) < 2:
                continue

            # HS6, HS10 추출
            hs6 = hs_code[:6] if hs_code and len(hs_code) >= 6 else ""
            hs10 = hs_code if hs_code and len(hs_code) >= 10 else ""

            sample_id = self._generate_id(product_name, hs4, i)

            samples.append(DataSample(
                id=sample_id,
                text=product_name,
                hs4=hs4,
                hs6=hs6,
                hs10=hs10,
                meta={
                    'original_index': i,
                    'case_number': case.get('case_number', ''),
                    'decision_date': case.get('decision_date', ''),
                }
            ))

        return samples

    def _stratified_split(
        self,
        samples: List[DataSample]
    ) -> Tuple[List[DataSample], List[DataSample], List[DataSample]]:
        """
        Stratified split 수행

        각 클래스에서 비율에 맞게 샘플 분배
        """
        random.seed(self.seed)

        # 클래스별 샘플 그룹화
        class_samples: Dict[str, List[DataSample]] = defaultdict(list)
        for sample in samples:
            class_samples[sample.hs4].append(sample)

        # 최소 샘플 수 필터링
        filtered_classes = {}
        dropped_count = 0
        for hs4, class_list in class_samples.items():
            if len(class_list) >= self.min_samples_per_class:
                filtered_classes[hs4] = class_list
            else:
                dropped_count += len(class_list)

        train_samples = []
        val_samples = []
        test_samples = []

        # 클래스별 분할
        for hs4, class_list in filtered_classes.items():
            random.shuffle(class_list)
            n = len(class_list)

            # 분할 인덱스 계산
            n_train = max(1, int(n * self.train_ratio))
            n_val = max(1, int(n * self.val_ratio))

            # Train은 최소 1개, 나머지 분배
            if n < 3:
                # 3개 미만이면 train에만
                train_samples.extend(class_list)
            else:
                train_samples.extend(class_list[:n_train])
                val_samples.extend(class_list[n_train:n_train + n_val])
                test_samples.extend(class_list[n_train + n_val:])

        # 셔플
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

        return train_samples, val_samples, test_samples

    def split(
        self,
        source_path: str,
        output_dir: str
    ) -> SplitStats:
        """
        데이터 분할 실행

        Args:
            source_path: 원본 데이터 경로
            output_dir: 출력 디렉토리

        Returns:
            분할 통계
        """
        print(f"[DataSplitter] 데이터 로드: {source_path}")

        # 데이터 로드
        samples = self._load_ruling_cases(source_path)
        print(f"  총 샘플: {len(samples)}")

        # 클래스 통계
        class_counts = Counter(s.hs4 for s in samples)
        total_classes = len(class_counts)
        valid_classes = sum(1 for c in class_counts.values() if c >= self.min_samples_per_class)

        print(f"  총 클래스: {total_classes}")
        print(f"  유효 클래스 (>={self.min_samples_per_class}샘플): {valid_classes}")

        # 분할
        train, val, test = self._stratified_split(samples)

        print(f"  Train: {len(train)}")
        print(f"  Val: {len(val)}")
        print(f"  Test: {len(test)}")

        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # JSONL로 저장
        def save_jsonl(data: List[DataSample], filename: str):
            filepath = output_path / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                for sample in data:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
            print(f"  저장: {filepath} ({len(data)}개)")

        save_jsonl(train, "hs4_train.jsonl")
        save_jsonl(val, "hs4_val.jsonl")
        save_jsonl(test, "hs4_test.jsonl")

        # 메타데이터 저장
        train_classes = set(s.hs4 for s in train)
        val_classes = set(s.hs4 for s in val)
        test_classes = set(s.hs4 for s in test)

        stats = SplitStats(
            total_samples=len(samples),
            train_samples=len(train),
            val_samples=len(val),
            test_samples=len(test),
            total_classes=total_classes,
            train_classes=len(train_classes),
            val_classes=len(val_classes),
            test_classes=len(test_classes),
            dropped_classes=total_classes - valid_classes,
            dropped_samples=len(samples) - len(train) - len(val) - len(test)
        )

        # splits.json 저장
        splits_meta = {
            "config": {
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "min_samples_per_class": self.min_samples_per_class,
                "seed": self.seed,
            },
            "stats": stats.to_dict(),
            "class_distribution": {
                "train": dict(Counter(s.hs4 for s in train)),
                "val": dict(Counter(s.hs4 for s in val)),
                "test": dict(Counter(s.hs4 for s in test)),
            },
            "files": {
                "train": "hs4_train.jsonl",
                "val": "hs4_val.jsonl",
                "test": "hs4_test.jsonl",
            }
        }

        splits_file = output_path / "splits.json"
        with open(splits_file, 'w', encoding='utf-8') as f:
            json.dump(splits_meta, f, ensure_ascii=False, indent=2)
        print(f"  저장: {splits_file}")

        return stats

    @staticmethod
    def load_split(split_path: str) -> List[DataSample]:
        """분할 파일 로드"""
        samples = []
        with open(split_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(DataSample.from_dict(json.loads(line)))
        return samples


def main():
    """CLI 엔트리포인트"""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="데이터 분할")
    parser.add_argument("--config", default="configs/benchmark.yaml", help="설정 파일")
    parser.add_argument("--source", help="원본 데이터 경로 (설정 파일 오버라이드)")
    parser.add_argument("--output", help="출력 디렉토리 (설정 파일 오버라이드)")
    args = parser.parse_args()

    # 설정 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_config = config.get('data', {})
    split_config = data_config.get('split', {})

    source_path = args.source or data_config.get('source_path', 'data/ruling_cases/all_cases_full_v7.json')
    output_dir = args.output or data_config.get('benchmark_dir', 'data/benchmarks')

    # 분할기 생성
    splitter = DataSplitter(
        train_ratio=split_config.get('train_ratio', 0.70),
        val_ratio=split_config.get('val_ratio', 0.15),
        test_ratio=split_config.get('test_ratio', 0.15),
        min_samples_per_class=split_config.get('min_samples_per_class', 3),
        seed=config.get('experiment', {}).get('seed', 42)
    )

    # 분할 실행
    stats = splitter.split(source_path, output_dir)

    print("\n[결과]")
    for k, v in stats.to_dict().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
