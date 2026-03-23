from pathlib import Path

from health_rag.loaders.markdown import MarkdownLoader


def test_markdown_loader_splits_headings(tmp_path: Path) -> None:
    p = tmp_path / "x.md"
    p.write_text(
        "# Title Here\n\n## First\n\nBody one.\n\n## Second\n\nBody two.\n",
        encoding="utf-8",
    )
    docs = MarkdownLoader().load(p)
    sections = {d.metadata["section"]: d.text.strip() for d in docs}
    assert "First" in sections and "one" in sections["First"]
    assert "Second" in sections and "two" in sections["Second"]
    assert all(d.metadata.get("path") for d in docs)
