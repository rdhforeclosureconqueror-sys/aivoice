from typing import List


def split_text_into_chunks(text: str, max_chars: int) -> List[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []

    if len(normalized) <= max_chars:
        return [normalized]

    words = normalized.split(" ")
    chunks: list[str] = []
    current = ""

    for word in words:
        if not current:
            if len(word) <= max_chars:
                current = word
            else:
                for i in range(0, len(word), max_chars):
                    chunks.append(word[i:i + max_chars])
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            chunks.append(current)
            if len(word) <= max_chars:
                current = word
            else:
                current = ""
                for i in range(0, len(word), max_chars):
                    chunks.append(word[i:i + max_chars])

    if current:
        chunks.append(current)

    return chunks
