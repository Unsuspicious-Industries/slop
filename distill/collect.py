from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from distill.config import DistillConfig, apply_model_preset, ensure_output_dirs
from distill.env import load_dotenv_if_present
from distill.prompts import create_prompt_variations, default_prompts, load_prompts
from distill.teachers import TeacherClient


MANIFEST_NAME = "manifest.jsonl"
SUMMARY_NAME = "summary.json"


def make_teacher(name: str, cfg: DistillConfig) -> TeacherClient:
    n = name.lower()
    if n == "grok":
        # Grok direct has no credits - use OpenRouter as fallback
        from distill.teachers.openrouter import OpenRouterTeacher
        return OpenRouterTeacher()
    if n == "openrouter":
        from distill.teachers.openrouter import OpenRouterTeacher
        return OpenRouterTeacher()
    if n == "openai":
        from distill.teachers.openai import OpenAITeacher

        partials = 2 if cfg.partial_supervision else 0
        return OpenAITeacher(partial_images=partials)
    if n == "dalle":
        from distill.teachers.dalle import DalleTeacher

        return DalleTeacher()
    if n == "openrouter":
        from distill.teachers.openrouter import OpenRouterTeacher

        return OpenRouterTeacher()
    raise ValueError(f"Unknown teacher: {name}")


def _load_prompt_list(cfg: DistillConfig) -> List[str]:
    if cfg.prompts_file:
        prompts = load_prompts(cfg.prompts_file, max_count=None)
    else:
        prompts = default_prompts(max_count=None)

    if cfg.auto_variations:
        prompts = create_prompt_variations(
            prompts,
            preset=cfg.variation_preset,
            max_count=None,
        )

    if cfg.max_samples is not None:
        prompts = prompts[: cfg.max_samples]
    return prompts


def _prompt_key(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:16]


def _image_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record_key(prompt_key: str, sample_index: int) -> str:
    return f"{prompt_key}:{sample_index:02d}"


def _read_manifest(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    records: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prompt_key = rec.get("prompt_key") or _prompt_key(rec["prompt"])
            sample_index = int(rec.get("sample_index", 0))
            records[_record_key(prompt_key, sample_index)] = rec
    return records


def _write_manifest(path: Path, records: Dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for key in sorted(records.keys()):
            f.write(json.dumps(records[key], ensure_ascii=False) + "\n")


def _write_summary(path: Path, cfg: DistillConfig, records: Dict[str, dict], started_at: float) -> None:
    unique_images = len({rec["final_sha256"] for rec in records.values()})
    unique_prompts = len({rec["prompt_key"] for rec in records.values()})
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": cfg.dataset_name,
                "run_id": cfg.run_id,
                "teacher": cfg.teacher,
                "model_id": cfg.model_id,
                "partial_supervision": cfg.partial_supervision,
                "num_records": len(records),
                "num_unique_prompts": unique_prompts,
                "num_unique_final_images": unique_images,
                "samples_per_prompt": cfg.samples_per_prompt,
                "auto_variations": cfg.auto_variations,
                "variation_preset": cfg.variation_preset,
                "started_at": started_at,
                "finished_at": time.time(),
            },
            f,
            indent=2,
        )


async def _collect_one(
    teacher: TeacherClient,
    prompt: str,
    out_dir: Path,
    idx: int,
    sample_index: int,
) -> dict:
    sample = await teacher.generate(prompt)
    prompt_key = _prompt_key(prompt)
    stem = f"{idx:06d}_{sample_index:02d}_{prompt_key}"

    final_path = out_dir / f"{stem}_final.png"
    final_path.write_bytes(sample.final_image)

    partial_paths: List[str] = []
    partial_sha256: List[str] = []
    for j, img in enumerate(sample.partial_images):
        p = out_dir / f"{stem}_partial{j}.png"
        p.write_bytes(img)
        partial_paths.append(str(p))
        partial_sha256.append(_image_sha256(img))

    return {
        "prompt": prompt,
        "prompt_key": prompt_key,
        "sample_index": sample_index,
        "teacher": sample.teacher,
        "final_path": str(final_path),
        "partial_paths": partial_paths,
        "final_sha256": _image_sha256(sample.final_image),
        "partial_sha256": partial_sha256,
        "partial_count": len(sample.partial_images),
        "metadata": sample.metadata,
        "collected_at": time.time(),
    }


async def collect(cfg: DistillConfig) -> Path:
    started_at = time.time()
    root = ensure_output_dirs(cfg)
    samples_dir = root / "samples"
    manifest_path = root / MANIFEST_NAME
    summary_path = root / SUMMARY_NAME

    prompts = _load_prompt_list(cfg)
    teacher = make_teacher(cfg.teacher, cfg)

    existing = {} if cfg.overwrite else _read_manifest(manifest_path)
    if cfg.overwrite and manifest_path.exists():
        manifest_path.unlink()

    existing_counts: Dict[str, int] = {}
    for rec in existing.values():
        key = rec["prompt_key"]
        existing_counts[key] = max(existing_counts.get(key, 0), int(rec.get("sample_index", 0)) + 1)

    pending: List[Tuple[int, int, str]] = []
    for i, prompt in enumerate(prompts):
        prompt_key = _prompt_key(prompt)
        start_j = existing_counts.get(prompt_key, 0) if cfg.reuse_existing else 0
        for sample_index in range(start_j, cfg.samples_per_prompt):
            pending.append((i, sample_index, prompt))

    sem = asyncio.Semaphore(cfg.max_concurrent)
    records = dict(existing)

    async def worker(i: int, sample_index: int, prompt: str):
        async with sem:
            rec = await _collect_one(teacher, prompt, samples_dir, i, sample_index)
            records[_record_key(rec["prompt_key"], sample_index)] = rec
            _write_manifest(manifest_path, records)
            _write_summary(summary_path, cfg, records, started_at)

    tasks = [asyncio.create_task(worker(i, sample_index, prompt)) for i, sample_index, prompt in pending]
    if tasks:
        await asyncio.gather(*tasks)
    else:
        _write_manifest(manifest_path, records)
        _write_summary(summary_path, cfg, records, started_at)

    return manifest_path


def export_prompts_csv(cfg: DistillConfig, output_csv: Path) -> None:
    prompts = _load_prompt_list(cfg)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt"])
        for prompt in prompts:
            writer.writerow([prompt])


def main():
    import argparse

    load_dotenv_if_present()

    parser = argparse.ArgumentParser(description="Collect teacher samples for distillation")
    parser.add_argument("--run-id", default="distill_run")
    parser.add_argument("--dataset-name", default="default")
    parser.add_argument("--teacher", default="grok", choices=["grok", "openai", "dalle"])
    parser.add_argument("--model-preset", default="sd15", choices=["sd15", "sd21", "sdxl"])
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--partial-supervision", action="store_true")
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--no-auto-variations", action="store_true")
    parser.add_argument("--variation-preset", default="bias_lite")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--export-prompts-csv", type=Path, default=None)
    args = parser.parse_args()

    cfg = DistillConfig(
        run_id=args.run_id,
        dataset_name=args.dataset_name,
        teacher=args.teacher,
        model_preset=args.model_preset,
        prompts_file=args.prompts_file,
        max_samples=args.max_samples,
        partial_supervision=args.partial_supervision,
        samples_per_prompt=args.samples_per_prompt,
        auto_variations=not args.no_auto_variations,
        variation_preset=args.variation_preset,
        max_concurrent=args.max_concurrent,
        overwrite=args.overwrite,
        reuse_existing=not args.no_reuse,
    )
    cfg = apply_model_preset(cfg)

    if args.export_prompts_csv is not None:
        export_prompts_csv(cfg, args.export_prompts_csv)
        return

    manifest = asyncio.run(collect(cfg))
    print(manifest)


if __name__ == "__main__":
    main()
