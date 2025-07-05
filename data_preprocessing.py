import os
import csv
import json
import re
import unicodedata
import sys
import time
from collections import Counter
from threading import Thread

csv.field_size_limit(10000000)

# 1. Token normalization
def clean_token(token):
    token = unicodedata.normalize("NFC", token)
    token = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", token)
    token = re.sub(r"\s+", " ", token).strip()
    token = token.replace("–", "-").replace("—", "-")
    token = token.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return token

# 2. Read and robustly parse HIPE TSV, handling NoSpaceAfter
def read_hipes_tsv(tsv_path, skipped_log=None):
    with open(tsv_path, encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#") and line.strip()]
    if not lines:
        return []
    reader = csv.reader(lines, delimiter="\t")
    headers = next(reader)
    headers_lc = [h.lower() for h in headers]
    try:
        token_idx = headers_lc.index("token")
        coarse_idx = headers_lc.index("ne-coarse-lit")
        fine_idx = headers_lc.index("ne-fine-lit") if "ne-fine-lit" in headers_lc else None
        misc_idx = headers_lc.index("misc") if "misc" in headers_lc else None
    except ValueError as e:
        raise ValueError(f"Missing expected column in {tsv_path}: {e}")
    ncols = len(headers)
    sentences = []
    current = []
    line_no = 2
    for row in reader:
        if len(row) != ncols:
            if skipped_log is not None:
                skipped_log.append({"file": tsv_path, "line": line_no, "reason": "wrong cols", "row": row})
            if current:
                sentences.append(current)
                current = []
            line_no += 1
            continue
        token = row[token_idx]
        if len(token) > 40 or " " in token:
            if skipped_log is not None:
                skipped_log.append({"file": tsv_path, "line": line_no, "reason": "not a token", "row": row})
            if current:
                sentences.append(current)
                current = []
            line_no += 1
            continue
        coarse_label = row[coarse_idx]
        fine_label = row[fine_idx] if fine_idx is not None else None
        misc_val = row[misc_idx] if misc_idx is not None else ""
        no_space_after = "NoSpaceAfter" in misc_val
        if token.strip() == "":
            if current:
                sentences.append(current)
                current = []
        else:
            token = clean_token(token)
            current.append((token, coarse_label, fine_label, no_space_after))
        line_no += 1
    if current:
        sentences.append(current)
    return sentences

# 3. Chunk by sentence (never split a sentence)
def chunk_by_sentence(sentences, max_tokens=384):
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_len = 0
        current_chunk.extend(sent)
        current_len += sent_len
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# 4. Output formatters with NoSpaceAfter handling
def chunk_to_docs(chunk):
    out = []
    for i, (token, _, _, no_space_after) in enumerate(chunk):
        out.append(token)
        if i < len(chunk) - 1:
            if not no_space_after:
                out.append(" ")
    return "".join(out)

def chunk_to_log(chunk):
    return "\n".join(token for token, _, _, _ in chunk)

def chunk_to_srt(chunk):
    srt_lines = []
    idx = 1
    pos = 0
    tokens_per_line = 10
    while pos < len(chunk):
        tokens = []
        for j in range(tokens_per_line):
            if pos + j < len(chunk):
                tokens.append(chunk[pos + j][0])
        s = " ".join(tokens)
        srt_lines.append(f"{idx}")
        srt_lines.append(f"00:00:{idx:02d},000 --> 00:00:{(idx+1):02d},000")
        srt_lines.append(s)
        srt_lines.append("")
        idx += 1
        pos += tokens_per_line
    return "\n".join(srt_lines)

# 5. Entity extraction with robust offset calculation and PERSON/ORG/LOC filter
def extract_entities(token_label_chunk, text):
    entities = []
    entity = None
    entity_types = {
        "PERSON": re.compile(r"^b-per(s)?$", re.I),
        "ORG": re.compile(r"^b-org(s)?$", re.I),
        "LOC": re.compile(r"^b-loc(s)?$", re.I)
    }
    i_types = {
        "PERSON": re.compile(r"^i-per(s)?$", re.I),
        "ORG": re.compile(r"^i-org(s)?$", re.I),
        "LOC": re.compile(r"^i-loc(s)?$", re.I)
    }
    tokens = [token for token, _, _, _ in token_label_chunk]
    no_space_afters = [no_space_after for _, _, _, no_space_after in token_label_chunk]
    token_start_positions = []
    offset = 0
    for i, token in enumerate(tokens):
        found = text.find(token, offset)
        if found == -1:
            token_start_positions.append(offset)
        else:
            token_start_positions.append(found)
            offset = found + len(token)
            if i < len(tokens)-1 and not no_space_afters[i]:
                offset += 1
    current_type = None
    for idx, (token, label, _, _) in enumerate(token_label_chunk):
        matched_type = next((etype for etype, rx in entity_types.items() if label and rx.match(label)), None)
        if matched_type:
            if entity:
                entity["end"] = token_start_positions[idx-1] + len(token_label_chunk[idx-1][0])
                entity["text"] = text[entity["start"]:entity["end"]]
                entities.append(entity)
            entity = {"start": token_start_positions[idx], "label": matched_type}
            current_type = matched_type
        elif current_type and label and i_types[current_type].match(label):
            pass
        else:
            if entity:
                entity["end"] = token_start_positions[idx-1] + len(token_label_chunk[idx-1][0])
                entity["text"] = text[entity["start"]:entity["end"]]
                entities.append(entity)
                entity = None
                current_type = None
    if entity:
        entity["end"] = token_start_positions[-1] + len(token_label_chunk[-1][0])
        entity["text"] = text[entity["start"]:entity["end"]]
        entities.append(entity)
    return [e for e in entities if e["label"] in ("PERSON", "ORG", "LOC")]

# 6. Output utilities
def ensure_dirs(base_out, formats):
    for fmt in formats:
        os.makedirs(os.path.join(base_out, fmt), exist_ok=True)
        os.makedirs(os.path.join(base_out, fmt + 'json'), exist_ok=True)

def process_tsv_file(
    tsv_path, base_out, max_tokens=384, skipped_log=None,
    hard_truncate=False, sliding_window=False, stride=256, log_dropped_entities=None
):
    basename = os.path.splitext(os.path.basename(tsv_path))[0]
    sentences = read_hipes_tsv(tsv_path, skipped_log=skipped_log)
    formats = ['docs', 'log', 'srt']

    if sliding_window:
        tokens = []
        for sent in sentences:
            tokens.extend(sent)
        total_tokens = len(tokens)
        chunk_idx = 0
        dropped_entities_log = []

        while chunk_idx * stride < total_tokens:
            start = chunk_idx * stride
            end = min(start + max_tokens, total_tokens)
            chunk = tokens[start:end]
            docs_text = chunk_to_docs(chunk)
            log_text = chunk_to_log(chunk)
            srt_text = chunk_to_srt(chunk)
            all_entities = extract_entities(chunk, docs_text)

            chunk_text = docs_text
            kept = []
            dropped = []
            for ent in all_entities:
                if 0 <= ent["start"] and ent["end"] <= len(chunk_text):
                    kept.append(ent)
                else:
                    dropped.append(ent)

            if log_dropped_entities is not None and dropped:
                dropped_entities_log.append({
                    "file": tsv_path, "chunk": chunk_idx,
                    "dropped_entities": dropped
                })

            out_docs = os.path.join(base_out, "docs", f"{basename}_{chunk_idx}.doc")
            with open(out_docs, "w", encoding="utf-8") as f:
                f.write(docs_text)
            out_log = os.path.join(base_out, "log", f"{basename}_{chunk_idx}.log")
            with open(out_log, "w", encoding="utf-8") as f:
                f.write(log_text)
            out_srt = os.path.join(base_out, "srt", f"{basename}_{chunk_idx}.srt")
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write(srt_text)
            for fmt, text in zip(formats, [docs_text, log_text, srt_text]):
                out_json = os.path.join(base_out, fmt + "json", f"{basename}_{chunk_idx}.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump({"entities": kept}, f, ensure_ascii=False, indent=2)
            chunk_idx += 1

        if log_dropped_entities is not None:
            log_dropped_entities.extend(dropped_entities_log)
    elif hard_truncate:
        tokens = []
        for sent in sentences:
            tokens.extend(sent)
        total_tokens = len(tokens)
        chunk_idx = 0
        dropped_entities_log = []

        while chunk_idx * max_tokens < total_tokens:
            start = chunk_idx * max_tokens
            end = min((chunk_idx+1)*max_tokens, total_tokens)
            chunk = tokens[start:end]
            docs_text = chunk_to_docs(chunk)
            log_text = chunk_to_log(chunk)
            srt_text = chunk_to_srt(chunk)
            all_entities = extract_entities(chunk, docs_text)
            chunk_text = docs_text
            entity_spans = []
            for ent in all_entities:
                entity_in_chunk = False
                if 0 <= ent["start"] < len(chunk_text) and 0 < ent["end"] <= len(chunk_text):
                    entity_in_chunk = True
                entity_spans.append((ent, entity_in_chunk))
            kept = [ent for ent, inside in entity_spans if inside]
            dropped = [ent for ent, inside in entity_spans if not inside]
            if log_dropped_entities is not None and dropped:
                dropped_entities_log.append({
                    "file": tsv_path, "chunk": chunk_idx,
                    "dropped_entities": dropped
                })
            out_docs = os.path.join(base_out, "docs", f"{basename}_{chunk_idx}.doc")
            with open(out_docs, "w", encoding="utf-8") as f:
                f.write(docs_text)
            out_log = os.path.join(base_out, "log", f"{basename}_{chunk_idx}.log")
            with open(out_log, "w", encoding="utf-8") as f:
                f.write(log_text)
            out_srt = os.path.join(base_out, "srt", f"{basename}_{chunk_idx}.srt")
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write(srt_text)
            for fmt, text in zip(formats, [docs_text, log_text, srt_text]):
                out_json = os.path.join(base_out, fmt + "json", f"{basename}_{chunk_idx}.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump({"entities": kept}, f, ensure_ascii=False, indent=2)
            chunk_idx += 1

        if log_dropped_entities is not None:
            log_dropped_entities.extend(dropped_entities_log)
    else:
        chunks = chunk_by_sentence(sentences, max_tokens=max_tokens)
        for chunk_idx, chunk in enumerate(chunks):
            docs_text = chunk_to_docs(chunk)
            log_text = chunk_to_log(chunk)
            srt_text = chunk_to_srt(chunk)
            entities = extract_entities(chunk, docs_text)
            out_docs = os.path.join(base_out, "docs", f"{basename}_{chunk_idx}.doc")
            with open(out_docs, "w", encoding="utf-8") as f:
                f.write(docs_text)
            out_log = os.path.join(base_out, "log", f"{basename}_{chunk_idx}.log")
            with open(out_log, "w", encoding="utf-8") as f:
                f.write(log_text)
            out_srt = os.path.join(base_out, "srt", f"{basename}_{chunk_idx}.srt")
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write(srt_text)
            for fmt, text in zip(formats, [docs_text, log_text, srt_text]):
                out_json = os.path.join(base_out, fmt + "json", f"{basename}_{chunk_idx}.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump({"entities": entities}, f, ensure_ascii=False, indent=2)

def batch_process(datadir, base_out, max_tokens=384, hard_truncate=False, sliding_window=False, stride=256):
    formats = ['docs', 'log', 'srt']
    ensure_dirs(base_out, formats)
    skipped_rows = []
    skipped_files = []
    dropped_entities = []
    for root, dirs, files in os.walk(datadir):
        for fname in files:
            if fname.endswith('.tsv'):
                tsv_path = os.path.join(root, fname)
                print(f"Processing {tsv_path}")
                try:
                    process_tsv_file(
                        tsv_path, base_out, max_tokens,
                        skipped_log=skipped_rows,
                        hard_truncate=hard_truncate,
                        sliding_window=sliding_window,
                        stride=stride,
                        log_dropped_entities=dropped_entities
                    )
                except Exception as e:
                    print(f"Skipping file {tsv_path} due to error: {e}")
                    skipped_files.append({
                        "file": tsv_path,
                        "error": str(e)
                    })
    if skipped_rows:
        skipped_log_path = os.path.join(base_out, "skipped_rows.log.json")
        with open(skipped_log_path, "w", encoding="utf-8") as logf:
            json.dump(skipped_rows, logf, ensure_ascii=False, indent=2)
        print(f"Skipped rows log written to {skipped_log_path}")
    if skipped_files:
        skipped_files_log_path = os.path.join(base_out, "skipped_files.log.json")
        with open(skipped_files_log_path, "w", encoding="utf-8") as logf:
            json.dump(skipped_files, logf, ensure_ascii=False, indent=2)
        print(f"Skipped files log written to {skipped_files_log_path}")
    if (hard_truncate or sliding_window) and dropped_entities:
        dropped_entities_path = os.path.join(base_out, "dropped_entities.log.json")
        with open(dropped_entities_path, "w", encoding="utf-8") as logf:
            json.dump(dropped_entities, logf, ensure_ascii=False, indent=2)
        print(f"Dropped entities log written to {dropped_entities_path}")

# Validation and merging entities for document-level evaluation
def merge_entities_from_windows(json_dir, basename, num_chunks):
    merged = []
    seen = set()
    for chunk_idx in range(num_chunks):
        fpath = os.path.join(json_dir, f"{basename}_{chunk_idx}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
            for ent in data.get("entities", []):
                key = (ent["label"], ent["text"], ent["start"], ent["end"])
                if key not in seen:
                    merged.append(ent)
                    seen.add(key)
    return merged

def count_chunks_for_basename(json_dir, basename):
    idx = 0
    while os.path.exists(os.path.join(json_dir, f"{basename}_{idx}.json")):
        idx += 1
    return idx

def count_files_and_entities(base_out):
    stats = {}
    for fmt in ['docs', 'log', 'srt']:
        path = os.path.join(base_out, fmt)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        stats[fmt] = {'total_files': len(files)}
        if fmt == 'docs':
            token_count = 0
            for f in files:
                with open(os.path.join(path, f), encoding="utf-8") as ff:
                    data = ff.read()
                    token_count += len(data.split())
            stats[fmt]['total_tokens'] = token_count
        if fmt == 'log':
            token_count = 0
            for f in files:
                with open(os.path.join(path, f), encoding="utf-8") as ff:
                    token_count += sum(1 for _ in ff if _.strip())
            stats[fmt]['total_tokens'] = token_count
        if fmt == 'srt':
            token_count = 0
            for f in files:
                with open(os.path.join(path, f), encoding="utf-8") as ff:
                    lines = ff.readlines()
                    for i, line in enumerate(lines):
                        if i % 4 == 2:
                            token_count += len(line.split())
            stats[fmt]['total_tokens'] = token_count
        fmt_json = fmt + 'json'
        path_json = os.path.join(base_out, fmt_json)
        json_files = [f for f in os.listdir(path_json) if os.path.isfile(os.path.join(path_json, f))]
        stats[fmt_json] = {'total_files': len(json_files)}
        entity_count = Counter()
        for f in json_files:
            with open(os.path.join(path_json, f), encoding="utf-8") as jf:
                try:
                    dat = json.load(jf)
                    for ent in dat.get("entities", []):
                        entity_count[ent["label"]] += 1
                except Exception:
                    pass
        stats[fmt_json]['total_entities'] = sum(entity_count.values())
        stats[fmt_json]['entities_by_label'] = dict(entity_count)
    return stats

def print_spinner(text, stop_event):
    spinner = ['|', '/', '-', '\\']
    idx = 0
    while not stop_event["stop"]:
        print(f"\r{text} {spinner[idx % len(spinner)]}", end='', flush=True)
        idx += 1
        time.sleep(0.1)
    print('\r' + ' ' * (len(text) + 2) + '\r', end='', flush=True)

def merge_entities_from_windows(json_dir, basename, num_chunks):
    merged = []
    seen = set()
    for chunk_idx in range(num_chunks):
        fpath = os.path.join(json_dir, f"{basename}_{chunk_idx}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
            for ent in data.get("entities", []):
                key = (ent["label"], ent["text"], ent["start"], ent["end"])
                if key not in seen:
                    merged.append(ent)
                    seen.add(key)
    return merged

def count_chunks_for_basename(json_dir, basename):
    idx = 0
    while os.path.exists(os.path.join(json_dir, f"{basename}_{idx}.json")):
        idx += 1
    return idx

def count_files_and_entities(base_out):
    stats = {}
    for fmt in ['docs', 'log', 'srt']:
        path = os.path.join(base_out, fmt)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        stats[fmt] = {'total_files': len(files)}
        if fmt == 'docs':
            token_count = 0
            for f in files:
                with open(os.path.join(path, f), encoding="utf-8") as ff:
                    data = ff.read()
                    token_count += len(data.split())
            stats[fmt]['total_tokens'] = token_count
        if fmt == 'log':
            token_count = 0
            for f in files:
                with open(os.path.join(path, f), encoding="utf-8") as ff:
                    token_count += sum(1 for _ in ff if _.strip())
            stats[fmt]['total_tokens'] = token_count
        if fmt == 'srt':
            token_count = 0
            for f in files:
                with open(os.path.join(path, f), encoding="utf-8") as ff:
                    lines = ff.readlines()
                    for i, line in enumerate(lines):
                        if i % 4 == 2:
                            token_count += len(line.split())
            stats[fmt]['total_tokens'] = token_count
        fmt_json = fmt + 'json'
        path_json = os.path.join(base_out, fmt_json)
        json_files = [f for f in os.listdir(path_json) if os.path.isfile(os.path.join(path_json, f))]
        stats[fmt_json] = {'total_files': len(json_files)}
        entity_count = Counter()
        for f in json_files:
            with open(os.path.join(path_json, f), encoding="utf-8") as jf:
                try:
                    dat = json.load(jf)
                    for ent in dat.get("entities", []):
                        entity_count[ent["label"]] += 1
                except Exception:
                    pass
        stats[fmt_json]['total_entities'] = sum(entity_count.values())
        stats[fmt_json]['entities_by_label'] = dict(entity_count)
    return stats

def validate_entities(datadir, base_out, max_tokens=384, sliding_window=False, stride=256):
    summary = []
    file_stats = []
    total_expected_entities = 0
    total_found_entities = 0
    total_missing = 0
    total_extra = 0
    files_with_errors = 0
    total_files = 0

    expected_label_counter = Counter()
    found_label_counter = Counter()
    missing_label_counter = Counter()
    extra_label_counter = Counter()

    skipped_files_path = os.path.join(base_out, "skipped_files.log.json")
    skipped_rows_path = os.path.join(base_out, "skipped_rows.log.json")
    skipped_files = []
    skipped_rows = []
    if os.path.exists(skipped_files_path):
        with open(skipped_files_path, "r", encoding="utf-8") as sf:
            skipped_files = json.load(sf)
    if os.path.exists(skipped_rows_path):
        with open(skipped_rows_path, "r", encoding="utf-8") as sr:
            skipped_rows = json.load(sr)

    stop_event = {"stop": False}
    spinner_thread = Thread(target=print_spinner, args=("Validating entities...", stop_event))
    spinner_thread.start()

    detailed_comparison = []

    for root, dirs, files in os.walk(datadir):
        for fname in files:
            if fname.endswith('.tsv'):
                total_files += 1
                tsv_path = os.path.join(root, fname)
                basename = os.path.splitext(os.path.basename(tsv_path))[0]
                try:
                    docsjson_dir = os.path.join(base_out, "docsjson")
                    num_chunks = count_chunks_for_basename(docsjson_dir, basename)
                    sentences = read_hipes_tsv(tsv_path)
                    tokens = []
                    for sent in sentences:
                        tokens.extend(sent)
                    total_tokens = len(tokens)
                    chunk_indices = []
                    if sliding_window:
                        chunk_idx = 0
                        while chunk_idx * stride < total_tokens:
                            start = chunk_idx * stride
                            end = min(start + max_tokens, total_tokens)
                            chunk_indices.append((start, end))
                            chunk_idx += 1
                    else:
                        s_chunks = chunk_by_sentence(sentences, max_tokens=max_tokens)
                        offset = 0
                        for chunk in s_chunks:
                            start = offset
                            end = offset + len(chunk)
                            chunk_indices.append((start, end))
                            offset = end

                    gold_entities = []
                    gold_seen = set()
                    for chunk_no, (start, end) in enumerate(chunk_indices):
                        chunk = tokens[start:end]
                        docs_text = chunk_to_docs(chunk)
                        ents = extract_entities(chunk, docs_text)
                        for ent in ents:
                            key = (ent["label"], ent["text"], ent["start"] + start, ent["end"] + start)
                            if key not in gold_seen:
                                ent2 = ent.copy()
                                ent2["start"] += start
                                ent2["end"] += start
                                gold_entities.append(ent2)
                                gold_seen.add(key)

                    found_entities = merge_entities_from_windows(docsjson_dir, basename, num_chunks)

                    gold_set = set((e["label"], e["text"], e["start"], e["end"]) for e in gold_entities)
                    found_set = set((e["label"], e["text"], e["start"], e["end"]) for e in found_entities)

                    missing = gold_set - found_set
                    extra = found_set - gold_set

                    for ent in gold_entities:
                        expected_label_counter[ent["label"]] += 1
                    for ent in found_entities:
                        found_label_counter[ent["label"]] += 1
                    for ent in missing:
                        missing_label_counter[ent[0]] += 1
                    for ent in extra:
                        extra_label_counter[ent[0]] += 1

                    total_expected_entities += len(gold_set)
                    total_found_entities += len(found_set)
                    total_missing += len(missing)
                    total_extra += len(extra)

                    file_stats.append({
                        "file": tsv_path,
                        "expected_entities": len(gold_set),
                        "found_entities": len(found_set),
                        "missing_entities": len(missing),
                        "extra_entities": len(extra),
                        "expected_entities_person": sum(1 for e in gold_entities if e["label"] == "PERSON"),
                        "expected_entities_org": sum(1 for e in gold_entities if e["label"] == "ORG"),
                        "expected_entities_loc": sum(1 for e in gold_entities if e["label"] == "LOC"),
                        "found_entities_person": sum(1 for e in found_entities if e["label"] == "PERSON"),
                        "found_entities_org": sum(1 for e in found_entities if e["label"] == "ORG"),
                        "found_entities_loc": sum(1 for e in found_entities if e["label"] == "LOC"),
                        "missing_entities_person": sum(1 for e in missing if e[0] == "PERSON"),
                        "missing_entities_org": sum(1 for e in missing if e[0] == "ORG"),
                        "missing_entities_loc": sum(1 for e in missing if e[0] == "LOC"),
                        "extra_entities_person": sum(1 for e in extra if e[0] == "PERSON"),
                        "extra_entities_org": sum(1 for e in extra if e[0] == "ORG"),
                        "extra_entities_loc": sum(1 for e in extra if e[0] == "LOC"),
                    })
                    detailed_comparison.append({
                        "file": tsv_path,
                        "expected": {
                            "PERSON": sum(1 for e in gold_entities if e["label"] == "PERSON"),
                            "ORG": sum(1 for e in gold_entities if e["label"] == "ORG"),
                            "LOC": sum(1 for e in gold_entities if e["label"] == "LOC"),
                        },
                        "found": {
                            "PERSON": sum(1 for e in found_entities if e["label"] == "PERSON"),
                            "ORG": sum(1 for e in found_entities if e["label"] == "ORG"),
                            "LOC": sum(1 for e in found_entities if e["label"] == "LOC"),
                        },
                        "missing": {
                            "PERSON": sum(1 for e in missing if e[0] == "PERSON"),
                            "ORG": sum(1 for e in missing if e[0] == "ORG"),
                            "LOC": sum(1 for e in missing if e[0] == "LOC"),
                        },
                        "extra": {
                            "PERSON": sum(1 for e in extra if e[0] == "PERSON"),
                            "ORG": sum(1 for e in extra if e[0] == "ORG"),
                            "LOC": sum(1 for e in extra if e[0] == "LOC"),
                        }
                    })

                    if missing or extra:
                        summary.append({
                            "file": tsv_path,
                            "num_missing": len(missing),
                            "num_extra": len(extra)
                        })
                except Exception as e:
                    files_with_errors += 1
                    summary.append({
                        "file": tsv_path,
                        "error": f"Validation error: {e}"
                    })
    stop_event["stop"] = True
    spinner_thread.join()

    output_stats = count_files_and_entities(base_out)

    print("\nValidation Summary (Document-Level, Sliding Window)")
    print("="*40)
    print(f"Total files processed:           {total_files}")
    print(f"Total files with errors:         {files_with_errors}")
    print(f"Total skipped files:             {len(skipped_files)}")
    print(f"Total skipped rows:              {len(skipped_rows)}")
    print(f"Total expected entities:         {total_expected_entities}")
    print(f"  PERSON: {expected_label_counter['PERSON']}")
    print(f"  ORG:    {expected_label_counter['ORG']}")
    print(f"  LOC:    {expected_label_counter['LOC']}")
    print(f"Total found entities:            {total_found_entities}")
    print(f"  PERSON: {found_label_counter['PERSON']}")
    print(f"  ORG:    {found_label_counter['ORG']}")
    print(f"  LOC:    {found_label_counter['LOC']}")
    print(f"Total missing entities:          {total_missing}")
    print(f"  PERSON: {missing_label_counter['PERSON']}")
    print(f"  ORG:    {missing_label_counter['ORG']}")
    print(f"  LOC:    {missing_label_counter['LOC']}")
    print(f"Total extra entities:            {total_extra}")
    print(f"  PERSON: {extra_label_counter['PERSON']}")
    print(f"  ORG:    {extra_label_counter['ORG']}")
    print(f"  LOC:    {extra_label_counter['LOC']}\n")
    print("=== Output breakdown by file type ===")
    for k, v in output_stats.items():
        print(f"{k:10s} : {v}")
    if summary:
        print(f"\n{len(summary)} file(s) had missing or extra entities. See validation_results.json for details.")
    if files_with_errors:
        print(f"\n{files_with_errors} file(s) had errors. See validation_results.json for details.")

    with open(os.path.join(base_out, "validation_results.json"), "w", encoding="utf-8") as vf:
        json.dump(summary, vf, ensure_ascii=False, indent=2)
    with open(os.path.join(base_out, "validation_stats.json"), "w", encoding="utf-8") as vf:
        json.dump(file_stats, vf, ensure_ascii=False, indent=2)
    with open(os.path.join(base_out, "validation_comparison_by_label.json"), "w", encoding="utf-8") as vf:
        json.dump({
            "expected": dict(expected_label_counter),
            "found": dict(found_label_counter),
            "missing": dict(missing_label_counter),
            "extra": dict(extra_label_counter),
            "detailed_comparison": detailed_comparison,
            "output_stats": output_stats
        }, vf, ensure_ascii=False, indent=2)
    print(f"\nSee validation_results.json, validation_stats.json, and validation_comparison_by_label.json in {base_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True, help="Directory containing .tsv files")
    parser.add_argument("--base_out", required=True, help="Output directory")
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--validate", action="store_true", help="Run validation after processing")
    parser.add_argument("--hard_truncate", action="store_true", help="Force hard truncation at max_tokens (may drop entities at boundaries)")
    parser.add_argument("--sliding_window", action="store_true", help="Use sliding window chunking to avoid entity split at boundaries")
    parser.add_argument("--stride", type=int, default=256, help="Stride for sliding window mode")
    args = parser.parse_args()
    batch_process(
        args.datadir, args.base_out, args.max_tokens,
        hard_truncate=args.hard_truncate,
        sliding_window=args.sliding_window,
        stride=args.stride
    )
    if args.validate:
        validate_entities(args.datadir, args.base_out, args.max_tokens, sliding_window=args.sliding_window, stride=args.stride)