import unicodedata
import re

def is_amharic(text, threshold=0.5):
    """Check if text contains enough Ge'ez script characters"""
    if not text or len(text.strip()) == 0:
        return False
    geez_chars = sum(1 for c in text if '\u1200' <= c <= '\u137F')
    return (geez_chars / len(text)) >= threshold

def normalize_amharic(text):
    """Normalize unicode — critical for Amharic"""
    # NFC normalization ensures consistent character representation
    text = unicodedata.normalize("NFC", text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove HTML artifacts if any
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def is_quality(amharic, english, min_chars=10, max_chars=500):
    """Filter out too-short or too-long pairs"""
    am_len = len(amharic.strip())
    en_len = len(english.strip())
    if am_len < min_chars or en_len < min_chars:
        return False
    if am_len > max_chars or en_len > max_chars:
        return False
    return True

# Run the full pipeline
print("Running preprocessing pipeline...")

clean_pairs = []
stats = {
    "total": len(amharic_lines),
    "failed_script": 0,
    "failed_quality": 0,
    "passed": 0
}

for am_line, en_line in zip(amharic_lines, english_lines):
    am = am_line.strip()
    en = en_line.strip()

    # Step 1: Normalize unicode
    am = normalize_amharic(am)
    en = en.strip()

    # Step 2: Script check — must be real Amharic
    if not is_amharic(am):
        stats["failed_script"] += 1
        continue

    # Step 3: Quality filter
    if not is_quality(am, en):
        stats["failed_quality"] += 1
        continue

    clean_pairs.append({"amharic": am, "english": en})
    stats["passed"] += 1

print(f"\n--- Pipeline Results ---")
print(f"Total input pairs:     {stats['total']:,}")
print(f"Failed script check:   {stats['failed_script']:,}")
print(f"Failed quality check:  {stats['failed_quality']:,}")
print(f"Clean pairs remaining: {stats['passed']:,}")
print(f"Retention rate:        {stats['passed']/stats['total']*100:.1f}%")

print(f"\n--- Sample cleaned pair ---")
print(f"EN: {clean_pairs[100]['english']}")
print(f"AM: {clean_pairs[100]['amharic']}")