from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from config import EncoderConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# semantic_model/ 目录
_PROJECT_ROOT = Path(__file__).parent.parent


class TextEncoder:

    def __init__(self, cfg: EncoderConfig | None = None) -> None:
        self.cfg = cfg or EncoderConfig()
        self._tokenizer = None
        self._model = None
        self._device: torch.device | None = None

    # ── 路径 ──────────────────────────────────────────────────────────────────

    def _model_dir(self) -> Path:
        """模型本地存储目录：semantic_model/models/{model_folder}/"""
        folder = self.cfg.model_name.split("/")[-1]
        return _PROJECT_ROOT / "models" / folder

    def _cache_pt(self, doc_id: str) -> Path:
        return _PROJECT_ROOT / "output" / "embeddings" / doc_id / "embeddings.pt"

    def _cache_meta(self, doc_id: str) -> Path:
        return _PROJECT_ROOT / "output" / "embeddings" / doc_id / "meta.json"

    # ── Hash ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _text_hash(texts: List[str]) -> str:
        """对文本列表计算 md5，文本有改动时缓存自动失效"""
        content = "\n".join(texts).encode("utf-8")
        return hashlib.md5(content).hexdigest()

    # ── 缓存 ─────────────────────────────────────────────────────────────────

    def _check_cache(
        self, doc_id: str, texts_hash: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """命中返回嵌入字典，未命中或校验失败返回 None"""
        pt = self._cache_pt(doc_id)
        meta_path = self._cache_meta(doc_id)
        if not pt.exists() or not meta_path.exists():
            return None
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("model_name") != self.cfg.model_name:
                return None
            if meta.get("hash") != texts_hash:
                return None
            return torch.load(pt, map_location="cpu", weights_only=True)
        except Exception:
            return None

    def _save_cache(
        self,
        doc_id: str,
        embeddings: Dict[str, torch.Tensor],
        texts_hash: str,
    ) -> None:
        pt = self._cache_pt(doc_id)
        pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, pt)
        meta = {
            "model_name": self.cfg.model_name,
            "hash": texts_hash,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self._cache_meta(doc_id), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ── 扫描 ─────────────────────────────────────────────────────────────────

    def _scan_one(
        self, graph_path: Path
    ) -> Tuple[str, List[str], List[str], str, Optional[Dict[str, torch.Tensor]]]:
        """读一个 graph.json，返回 (doc_id, event_ids, texts, hash, cached_or_None)"""
        with open(graph_path, encoding="utf-8") as f:
            data = json.load(f)
        doc_id = data["doc_id"]
        events = data["events"]
        event_ids = [e["event_id"] for e in events]
        texts = [e["text"] for e in events]
        h = self._text_hash(texts)
        cached = self._check_cache(doc_id, h)
        return doc_id, event_ids, texts, h, cached

    def _scan_all(
        self, graph_dir: Path
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        并发扫描所有 graph.json，返回 (待编码列表, 已缓存列表)。
        待编码：[{doc_id, event_ids, texts, hash}]
        已缓存：[{doc_id, embeddings}]
        """
        graph_paths = sorted(graph_dir.rglob("graph.json"))
        to_encode: List[Dict] = []
        cached_list: List[Dict] = []

        with ThreadPoolExecutor(max_workers=self.cfg.resolved_workers()) as pool:
            futures = {pool.submit(self._scan_one, p): p for p in graph_paths}
            for future in as_completed(futures):
                doc_id, event_ids, texts, h, cached = future.result()
                if cached is not None:
                    cached_list.append({"doc_id": doc_id, "embeddings": cached})
                else:
                    to_encode.append({
                        "doc_id": doc_id,
                        "event_ids": event_ids,
                        "texts": texts,
                        "hash": h,
                    })

        logger.info(
            "[扫描完成] 共 %d 个文档：%d 个已缓存，%d 个待编码",
            len(graph_paths), len(cached_list), len(to_encode),
        )
        return to_encode, cached_list

    # ── 模型加载 ──────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """懒加载：仅在首次编码时触发，全部命中缓存时不加载"""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModel

        model_dir = self._model_dir()
        weight_exists = (model_dir / "model.safetensors").exists() or \
                        (model_dir / "pytorch_model.bin").exists()
        if not weight_exists:
            logger.info("[下载] %s → %s", self.cfg.model_name, model_dir)
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=self.cfg.model_name,
                local_dir=str(model_dir),
                ignore_patterns=["*.h5", "*.msgpack", "flax_*", "tf_*", "rust_model.ot"],
            )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._model = AutoModel.from_pretrained(str(model_dir)).to(self._device).eval()
        dim = self._model.config.hidden_size
        logger.info("[模型加载] %s → %s (%dd)", self.cfg.model_name, self._device, dim)

    # ── 编码 ─────────────────────────────────────────────────────────────────

    def _encode_batch(
        self, to_encode: List[Dict]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        把所有待编码文档的文本拼成一个大 batch，统一 GPU forward，再按文档还原。
        """
        self._load_model()

        # 拼大列表，记录位置映射 (doc_id, event_id)
        all_texts: List[str] = []
        index_map: List[Tuple[str, str]] = []
        for doc in to_encode:
            for eid, text in zip(doc["event_ids"], doc["texts"]):
                all_texts.append(text)
                index_map.append((doc["doc_id"], eid))

        total_texts = len(all_texts)
        bs = self.cfg.batch_size
        total_batches = (total_texts + bs - 1) // bs

        assert self._tokenizer is not None and self._model is not None
        all_cls: List[torch.Tensor] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, total_texts, bs),
                total=total_batches,
                desc="编码中",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} 批  [剩余 {remaining}]",
            ):
                batch = all_texts[i: i + bs]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                all_cls.append(outputs.last_hidden_state[:, 0, :].cpu())

        all_cls_tensor = torch.cat(all_cls, dim=0)  # [total_texts, dim]

        # 还原到 {doc_id: {event_id: Tensor}}
        results: Dict[str, Dict[str, torch.Tensor]] = {}
        for idx, (doc_id, event_id) in enumerate(index_map):
            results.setdefault(doc_id, {})[event_id] = all_cls_tensor[idx]
        return results

    # ── 缓存写入 ──────────────────────────────────────────────────────────────

    def _write_caches(
        self,
        encoded: Dict[str, Dict[str, torch.Tensor]],
        hash_map: Dict[str, str],
    ) -> None:
        """并发写入所有新编码文档的缓存"""
        def _write(doc_id: str) -> None:
            self._save_cache(doc_id, encoded[doc_id], hash_map[doc_id])

        with ThreadPoolExecutor(max_workers=self.cfg.resolved_workers()) as pool:
            list(pool.map(_write, encoded.keys()))

    # ── 主入口 ───────────────────────────────────────────────────────────────

    def encode_all(
        self, graph_dir: str | Path
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        扫描 graph_dir 下所有 graph.json，返回全部文档的节点嵌入。

        Returns:
            {doc_id: {event_id: Tensor[dim]}}
        """
        graph_dir = Path(graph_dir)
        to_encode, cached_list = self._scan_all(graph_dir)

        encoded: Dict[str, Dict[str, torch.Tensor]] = {}
        if to_encode:
            encoded = self._encode_batch(to_encode)
            hash_map = {doc["doc_id"]: doc["hash"] for doc in to_encode}
            self._write_caches(encoded, hash_map)

        cached_results = {item["doc_id"]: item["embeddings"] for item in cached_list}
        all_results = {**cached_results, **encoded}

        total_nodes = sum(len(v) for v in all_results.values())
        logger.info(
            "[完成] %d 个文档，%d 个新编码，%d 个缓存加载，共 %d 个节点嵌入",
            len(all_results), len(encoded), len(cached_list), total_nodes,
        )
        return all_results


if __name__ == "__main__":
    cfg = EncoderConfig()
    graph_dir = _PROJECT_ROOT / "data" / "output" / "doc_results"

    encoder = TextEncoder(cfg)
    results = encoder.encode_all(graph_dir)

    # 打印每个文档的节点数和嵌入维度
    print("\n=== 编码结果摘要 ===")
    for doc_id, node_embs in sorted(results.items()):
        first_tensor = next(iter(node_embs.values()))
        print(f"  {doc_id}: {len(node_embs)} 个节点，嵌入维度 {first_tensor.shape[0]}")
