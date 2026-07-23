import hashlib


CHUNK_ID_PREFIX = "chunk_"
CHUNK_ID_SCHEME = "sha256-v1"


def create_chunk_id(source_url: str, strategy: str, text: str) -> str:
    """Create a stable content-addressed ID for a chunk."""
    identity = "\n".join((source_url, strategy, text))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"{CHUNK_ID_PREFIX}{digest}"


def is_stable_chunk_id(chunk_id: object) -> bool:
    """Return whether a value matches the stable chunk-ID format."""
    if not isinstance(chunk_id, str) or not chunk_id.startswith(CHUNK_ID_PREFIX):
        return False

    digest = chunk_id.removeprefix(CHUNK_ID_PREFIX)
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
