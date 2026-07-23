from pathlib import Path

import pytest

from scripts.docs.build_wiki import rewrite_markdown


def test_rewrite_markdown_maps_known_pages_and_local_images(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "guide.md"
    target = docs / "api.md"
    image = docs / "architecture.svg"
    source.write_text("", encoding="utf-8")
    target.write_text("", encoding="utf-8")
    image.write_text("<svg/>", encoding="utf-8")

    rendered = rewrite_markdown(
        source,
        "[API](api.md#types)\n![Architecture](architecture.svg)\n",
        {target.resolve(): "API-Reference.md"},
    )

    assert "[API](API-Reference.md#types)" in rendered
    assert "![Architecture](images/architecture.svg)" in rendered


def test_rewrite_markdown_strips_unknown_repo_links_but_keeps_external(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    rendered = rewrite_markdown(
        source,
        "[Missing](missing.md) [PyTorch](https://pytorch.org)\n",
        {},
    )

    assert "Missing" in rendered and "missing.md" not in rendered
    assert "[PyTorch](https://pytorch.org)" in rendered


def test_rewrite_markdown_rejects_existing_unmapped_document(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "orphan.md"
    source.write_text("", encoding="utf-8")
    target.write_text("# Orphan\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unmapped local documentation link"):
        rewrite_markdown(source, "[Orphan](orphan.md)\n", {})
