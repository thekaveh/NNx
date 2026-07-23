from pathlib import Path

import pytest
import yaml

import scripts.docs.build_docs as build_docs
from scripts.docs.build_docs import load_sections, render_mkdocs, render_site, render_wiki, validate_links


def test_manifest_sources_exist_and_numbered_labels_are_unique():
    pages = [page for _, group in load_sections() for page in group]
    assert pages
    assert all(page.source.is_file() for page in pages)
    assert len({page.number for page in pages}) == len(pages)
    assert len({page.slug for page in pages}) == len(pages)
    assert len({page.source.resolve() for page in pages}) == len(pages)


def test_generated_mkdocs_has_no_repository_chrome(tmp_path: Path):
    output = tmp_path / "mkdocs.yml"
    render_mkdocs(output)
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["docs_dir"] == "generated/site"
    assert "repo_url" not in config
    assert "repo_name" not in config
    labels = str(config["nav"])
    assert "1. Home" in labels
    assert "21. License" in labels
    assert {key: config[key] for key in ("site_name", "site_description", "site_url", "docs_dir", "site_dir")} == {
        "site_name": "NNx",
        "site_description": "Lightweight PyTorch training, evaluation, and visualization toolkit",
        "site_url": "https://thekaveh.github.io/NNx/",
        "docs_dir": "generated/site",
        "site_dir": "site",
    }
    assert config["theme"] == {
        "name": "material",
        "font": {"text": "Inter", "code": "JetBrains Mono"},
        "features": ["navigation.sections", "navigation.expand", "content.code.copy", "search.suggest"],
        "palette": [
            {"scheme": "slate", "toggle": {"icon": "material/weather-sunny", "name": "Switch to light mode"}},
            {"scheme": "default", "toggle": {"icon": "material/weather-night", "name": "Switch to dark mode"}},
        ],
    }
    assert config["extra_css"] == ["stylesheets/extra.css"]
    assert config["plugins"] == [
        "search",
        {
            "mkdocstrings": {
                "handlers": {
                    "python": {
                        "options": {
                            "docstring_style": "google",
                            "show_source": True,
                            "show_root_heading": True,
                            "show_signature_annotations": True,
                        }
                    }
                }
            }
        },
    ]
    assert config["markdown_extensions"] == [
        "admonition",
        "attr_list",
        "pymdownx.details",
        "pymdownx.superfences",
        "pymdownx.highlight",
        "pymdownx.inlinehilite",
        {"toc": {"permalink": True}},
    ]
    assert config["nav"] == [
        {"1. Home": "index.md"},
        {"2. Quickstart": "Quickstart.md"},
        {
            "Core guides": [
                {"3. Concepts": "Concepts.md"},
                {"4. Model surgery": "Model-surgery.md"},
                {"5. HuggingFace Hub": "HuggingFace-Hub.md"},
                {"6. Embeddings": "Embeddings.md"},
                {"7. Language modeling": "Language-modeling.md"},
                {"8. I-JEPA": "I-JEPA.md"},
                {"9. Experimental GGUF export": "Experimental-GGUF-export.md"},
                {"10. DPO": "DPO.md"},
            ]
        },
        {
            "Reference": [
                {"11. API reference": "API-reference.md"},
                {"12. External dependency contracts": "External-dependency-contracts.md"},
                {"13. Framework comparison": "Framework-comparison.md"},
                {"14. Architecture": "Architecture.md"},
            ]
        },
        {
            "Project": [
                {"15. Repository overview": "Repository-Overview.md"},
                {"16. Examples": "Examples.md"},
                {"17. Contributing": "Contributing.md"},
                {"18. Security policy": "Security-Policy.md"},
                {"19. Test import boundaries": "Test-Import-Boundaries.md"},
                {"20. Changelog": "Changelog.md"},
                {"21. License": "License.md"},
            ]
        },
    ]


def test_manifest_rejects_nonsequential_numbering(tmp_path: Path, monkeypatch):
    manifest = yaml.safe_load(build_docs.MANIFEST.read_text(encoding="utf-8"))
    manifest["sections"][1]["number"] = "22"
    invalid = tmp_path / "manifest.yaml"
    invalid.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    monkeypatch.setattr(build_docs, "MANIFEST", invalid)

    with pytest.raises(ValueError, match="ordered sequence"):
        build_docs.load_sections()


@pytest.mark.parametrize("contents", ["", "- sections"])
def test_manifest_rejects_non_mapping_root(tmp_path: Path, monkeypatch, contents: str):
    invalid = tmp_path / "manifest.yaml"
    invalid.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(build_docs, "MANIFEST", invalid)

    with pytest.raises(ValueError, match="invalid documentation manifest"):
        build_docs.load_sections()


@pytest.mark.parametrize(
    "sections",
    [
        [None],
        [{"title": "Group", "children": "not-a-list"}],
        [{"number": "1", "title": "Missing source"}],
    ],
)
def test_manifest_rejects_malformed_nested_schema(tmp_path: Path, monkeypatch, sections):
    invalid = tmp_path / "manifest.yaml"
    invalid.write_text(
        yaml.safe_dump({"surfaces": ["repo", "site", "wiki"], "numbering": "navigation", "sections": sections}),
        encoding="utf-8",
    )
    monkeypatch.setattr(build_docs, "MANIFEST", invalid)

    with pytest.raises(ValueError, match="invalid documentation manifest"):
        build_docs.load_sections()


def test_manifest_rejects_empty_group(tmp_path: Path, monkeypatch):
    invalid = tmp_path / "manifest.yaml"
    invalid.write_text(
        yaml.safe_dump(
            {
                "surfaces": ["repo", "site", "wiki"],
                "numbering": "navigation",
                "sections": [{"title": "Empty", "children": []}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(build_docs, "MANIFEST", invalid)

    with pytest.raises(ValueError, match="non-empty children"):
        build_docs.load_sections()


def test_manifest_rejects_unsafe_slug(tmp_path: Path, monkeypatch):
    manifest = yaml.safe_load(build_docs.MANIFEST.read_text(encoding="utf-8"))
    manifest["sections"][0]["slug"] = "../escape"
    invalid = tmp_path / "manifest.yaml"
    invalid.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    monkeypatch.setattr(build_docs, "MANIFEST", invalid)

    with pytest.raises(ValueError, match="filename-safe slug"):
        build_docs.load_sections()


@pytest.mark.parametrize(("location", "key"), [("root", "sectons"), ("section", "titel"), ("page", "slgu")])
def test_manifest_rejects_unknown_keys(tmp_path: Path, monkeypatch, location: str, key: str):
    manifest = yaml.safe_load(build_docs.MANIFEST.read_text(encoding="utf-8"))
    if location == "root":
        manifest[key] = []
    elif location == "section":
        manifest["sections"][2][key] = "typo"
    else:
        manifest["sections"][2]["children"][0][key] = "typo"
    invalid = tmp_path / "manifest.yaml"
    invalid.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    monkeypatch.setattr(build_docs, "MANIFEST", invalid)

    with pytest.raises(ValueError, match="unknown keys"):
        build_docs.load_sections()


@pytest.mark.parametrize("slug", ["index", "INDEX"])
def test_manifest_rejects_site_output_collisions(tmp_path: Path, monkeypatch, slug: str):
    manifest = yaml.safe_load(build_docs.MANIFEST.read_text(encoding="utf-8"))
    manifest["sections"][1]["slug"] = slug
    invalid = tmp_path / "manifest.yaml"
    invalid.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    monkeypatch.setattr(build_docs, "MANIFEST", invalid)

    with pytest.raises(ValueError, match="site output paths"):
        build_docs.load_sections()


def test_output_rejects_repository_root():
    with pytest.raises(ValueError, match="repository or one of its ancestors"):
        render_wiki(build_docs.ROOT)


def test_output_rejects_symlink(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        render_wiki(link)


def test_output_rejects_symlinked_parent(tmp_path: Path, monkeypatch):
    external = tmp_path / "external"
    target = external / "site"
    target.mkdir(parents=True)
    sentinel = target / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    generated = tmp_path / "generated"
    generated.symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(build_docs, "GENERATED", generated)

    with pytest.raises(ValueError, match="symlink"):
        render_site(generated / "site")

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_output_accepts_standard_macos_tmp_alias(monkeypatch):
    output = Path("/tmp/nnx-docs-output-portability-test")
    if not Path("/tmp").is_symlink() or Path("/tmp").resolve() != Path("/private/tmp"):
        pytest.skip("macOS /tmp alias is not present")
    monkeypatch.setattr(build_docs, "ROOT", Path("/Users/nnx/repository"))
    build_docs._prepare_output(output)
    output.rmdir()


def test_output_accepts_standard_macos_var_alias(monkeypatch):
    output = Path("/var/tmp/nnx-docs-output-portability-test")
    if not Path("/var").is_symlink() or Path("/var").resolve() != Path("/private/var"):
        pytest.skip("macOS /var alias is not present")
    monkeypatch.setattr(build_docs, "ROOT", Path("/Users/nnx/repository"))
    build_docs._prepare_output(output)
    output.rmdir()


def test_output_rejects_nonempty_unmanaged_directory_without_deleting_it(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "unrelated.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="nonempty unmanaged"):
        render_wiki(output)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_check_mode_does_not_replace_generated_outputs(tmp_path: Path, monkeypatch):
    generated = tmp_path / "generated"
    site = generated / "site"
    site.mkdir(parents=True)
    sentinel = site / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(build_docs, "GENERATED", generated)

    build_docs.build(check=True)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_site_and_wiki_are_self_contained(tmp_path: Path):
    site = tmp_path / "site"
    wiki = tmp_path / "wiki"
    render_site(site)
    render_wiki(wiki)
    validate_links(site, "site")
    validate_links(wiki, "wiki")
    for root in (site, wiki):
        markdown = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.md"))
        assert "](https://github.com/thekaveh/NNx/blob/" not in markdown
    assert (site / "assets" / "training-lifecycle.svg").is_file()
    assert (wiki / "images" / "docs-projection.svg").is_file()
    assert "(Concepts)" in (wiki / "Home.md").read_text(encoding="utf-8")
    assert "(Concepts.md)" not in (wiki / "Home.md").read_text(encoding="utf-8")
    assert "security/advisories/new" in (wiki / "Security-Policy.md").read_text(encoding="utf-8")


def test_fragment_only_links_are_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Present\n\n[bad](#missing)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken site anchor"):
        validate_links(tmp_path, "site")


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_links_inside_fenced_examples_are_not_validated(tmp_path: Path, fence: str):
    (tmp_path / "Page.md").write_text(f"{fence}markdown\n[example](missing.md)\n{fence}\n", encoding="utf-8")
    validate_links(tmp_path, "site")


@pytest.mark.parametrize("delimiter", ["`", "``"])
def test_links_inside_inline_code_are_not_validated(tmp_path: Path, delimiter: str):
    (tmp_path / "Page.md").write_text(
        f"Use {delimiter}[example](missing.md){delimiter} literally.\n",
        encoding="utf-8",
    )
    validate_links(tmp_path, "site")


def test_nested_inline_link_syntax_is_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("[outer [inner]](Missing(a(b)c).md)\n", encoding="utf-8")

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


def test_escaped_opening_bracket_is_not_validated_as_a_link(tmp_path: Path):
    (tmp_path / "Page.md").write_text("\\[example](Missing.md)\n", encoding="utf-8")

    validate_links(tmp_path, "site")


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_info_like_fence_content_does_not_end_validation_fence(tmp_path: Path, fence: str):
    (tmp_path / "Page.md").write_text(
        f"{fence}text\n{fence}python\n[example](missing.md)\n{fence}\n",
        encoding="utf-8",
    )
    validate_links(tmp_path, "site")


def test_wiki_duplicate_heading_anchors_use_github_suffixes(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Changed\n\n# Changed\n\n[second](#changed-1)\n", encoding="utf-8")
    validate_links(tmp_path, "wiki")


def test_wiki_heading_anchor_allocation_avoids_global_collisions(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        "# Foo\n\n# Foo\n\n# Foo-1\n\n[third](#foo-1-1)\n",
        encoding="utf-8",
    )

    validate_links(tmp_path, "wiki")


@pytest.mark.parametrize("surface", ["site", "wiki"])
def test_setext_heading_anchors_are_validated(tmp_path: Path, surface: str):
    (tmp_path / "Page.md").write_text(
        "Setext heading\n==============\n\n[heading](#setext-heading)\n", encoding="utf-8"
    )

    validate_links(tmp_path, surface)


def test_explicit_attr_list_heading_ids_are_validated_for_site(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Descriptive heading {#stable-id}\n\n[heading](#stable-id)\n", encoding="utf-8")

    validate_links(tmp_path, "site")


def test_combined_attr_list_heading_ids_are_validated_for_site(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        '# Descriptive heading {#stable-id .featured data-kind="guide"}\n\n[heading](#stable-id)\n',
        encoding="utf-8",
    )

    validate_links(tmp_path, "site")


@pytest.mark.parametrize(
    ("heading", "anchor"),
    [
        ("# A  B", "a-b"),
        ("# A &amp; B", "a-b"),
    ],
)
def test_automatic_heading_ids_match_python_markdown(tmp_path: Path, heading: str, anchor: str):
    (tmp_path / "Page.md").write_text(f"{heading}\n\n[heading](#{anchor})\n", encoding="utf-8")

    validate_links(tmp_path, "site")


def test_wiki_automatic_heading_ids_decode_entities(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# A &amp; B\n\n[heading](#a-b)\n", encoding="utf-8")

    validate_links(tmp_path, "wiki")


@pytest.mark.parametrize(
    ("heading", "anchor"),
    [("# [Release notes](https://example.com)", "release-notes"), ("# Über", "über")],
)
def test_wiki_heading_ids_use_rendered_github_text(tmp_path: Path, heading: str, anchor: str):
    (tmp_path / "Page.md").write_text(f"{heading}\n\n[heading](#{anchor})\n", encoding="utf-8")

    validate_links(tmp_path, "wiki")


def test_links_inside_blockquoted_fences_are_not_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        "> ~~~markdown\n> [example](Missing.md)\n> [missing]: Missing.md\n> ~~~\n",
        encoding="utf-8",
    )

    validate_links(tmp_path, "wiki")


def test_links_inside_pre_blocks_are_not_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("<pre>\n[example](Missing.md)\n</pre>\n", encoding="utf-8")

    validate_links(tmp_path, "site")


def test_raw_html_links_and_images_are_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        '<a href="Missing.md">Missing</a> <img src="missing.png">\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


def test_unquoted_raw_html_links_are_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("<a href=Missing.md>Missing</a>\n", encoding="utf-8")

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


def test_markdown_like_html_attributes_are_not_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        '<table data-example="[Literal](Missing.md)"><tr><td>ok</td></tr></table>\n',
        encoding="utf-8",
    )

    validate_links(tmp_path, "site")


def test_links_inside_list_contained_fences_are_not_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("- ~~~markdown\n  [Literal](Missing.md)\n  ~~~\n", encoding="utf-8")

    validate_links(tmp_path, "wiki")


def test_list_contained_heading_anchors_are_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("- ## Listed heading\n\n[heading](#listed-heading)\n", encoding="utf-8")

    validate_links(tmp_path, "wiki")


def test_site_does_not_create_anchors_from_superfence_contents(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        "```markdown\n# Phantom heading\n```\n\n[phantom](#phantom-heading)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="broken site anchor"):
        validate_links(tmp_path, "site")


@pytest.mark.parametrize(
    "text",
    [
        '```html\n<a href="Missing.md">literal</a>\n```\n',
        '<!-- <a href="Missing.md">literal</a> -->\n',
        '<pre><a href="Missing.md">literal</a></pre>\n',
    ],
)
def test_html_targets_in_literal_blocks_are_not_validated(tmp_path: Path, text: str):
    (tmp_path / "Page.md").write_text(text, encoding="utf-8")

    validate_links(tmp_path, "wiki")


def test_literal_html_anchor_in_fence_does_not_satisfy_wiki_fragment(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        '```html\n<a id="phantom"></a>\n```\n\n[phantom](#phantom)\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="broken wiki anchor"):
        validate_links(tmp_path, "wiki")


def test_manifest_rejects_duplicate_mapping_keys(tmp_path: Path, monkeypatch):
    duplicate = tmp_path / "manifest.yaml"
    duplicate.write_text(
        "surfaces: [repo, site, wiki]\nnumbering: navigation\nsections:\n"
        "  - number: '1'\n    title: Home\n    source: README.md\n    slug: Home\n    slug: Duplicate\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(build_docs, "MANIFEST", duplicate)

    with pytest.raises(ValueError, match="duplicate key"):
        build_docs.load_sections()


def test_link_validation_rejects_projection_escape(tmp_path: Path):
    projection = tmp_path / "projection"
    projection.mkdir()
    (tmp_path / "Outside.md").write_text("# Outside\n", encoding="utf-8")
    (projection / "Page.md").write_text("[outside](../Outside.md)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the projection"):
        validate_links(projection, "wiki")


def test_site_anchor_discovery_matches_configured_extensions(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        "??? note\n    ## Detail heading\n\n[detail](#detail-heading)\n",
        encoding="utf-8",
    )
    validate_links(tmp_path, "site")


def test_site_anchor_discovery_accepts_id_on_any_html_element(tmp_path: Path):
    (tmp_path / "Page.md").write_text('<div id="stable"></div>\n\n[stable](#stable)\n', encoding="utf-8")
    validate_links(tmp_path, "site")


def test_source_map_rejects_asset_symlinks(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repo"
    assets = repository / "docs" / "assets"
    assets.mkdir(parents=True)
    outside = tmp_path / "outside.svg"
    outside.write_text("<svg/>", encoding="utf-8")
    (assets / "escape.svg").symlink_to(outside)
    (repository / "README.md").write_text("# Home\n", encoding="utf-8")
    (repository / "docs" / "manifest.yaml").write_text(
        "surfaces: [repo, site, wiki]\nnumbering: navigation\nsections:\n"
        "  - number: '1'\n    title: Home\n    source: README.md\n    slug: Home\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(build_docs, "ROOT", repository)
    monkeypatch.setattr(build_docs, "MANIFEST", repository / "docs" / "manifest.yaml")
    with pytest.raises(ValueError, match="asset symlink"):
        build_docs._source_map("site")


def test_source_map_rejects_asset_directory_symlinks(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repo"
    assets = repository / "docs" / "assets"
    assets.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "escape.svg").write_text("<svg/>", encoding="utf-8")
    (assets / "escape").symlink_to(outside, target_is_directory=True)
    (repository / "README.md").write_text("# Home\n", encoding="utf-8")
    manifest = repository / "docs" / "manifest.yaml"
    manifest.write_text(
        "surfaces: [repo, site, wiki]\nnumbering: navigation\nsections:\n"
        "  - number: '1'\n    title: Home\n    source: README.md\n    slug: Home\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(build_docs, "ROOT", repository)
    monkeypatch.setattr(build_docs, "MANIFEST", manifest)
    with pytest.raises(ValueError, match="asset symlink"):
        build_docs._source_map("site")


def test_wiki_nested_list_heading_anchor_is_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        "- item\n    - ## Nested heading\n\n[heading](#nested-heading)\n",
        encoding="utf-8",
    )
    validate_links(tmp_path, "wiki")


def test_wiki_list_thematic_break_does_not_create_setext_anchor(tmp_path: Path):
    (tmp_path / "Page.md").write_text("- item\n  ---\n\n[item](#item)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken wiki anchor"):
        validate_links(tmp_path, "wiki")


@pytest.mark.parametrize("tag", ["script", "style", "div"])
def test_markdown_inside_raw_html_blocks_is_not_validated(tmp_path: Path, tag: str):
    (tmp_path / "Page.md").write_text(f"<{tag}>\n[Literal](Missing.md)\n</{tag}>\n", encoding="utf-8")

    validate_links(tmp_path, "site")


def test_blockquoted_indented_code_is_not_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text(">\n>     [Literal](Missing.md)\n", encoding="utf-8")

    validate_links(tmp_path, "wiki")


def test_blockquoted_reference_definitions_are_validated(tmp_path: Path):
    (tmp_path / "Target.md").write_text("# Target\n", encoding="utf-8")
    (tmp_path / "Page.md").write_text(
        "> [Target][ref]\n>\n> [ref]: Target.md#target\n",
        encoding="utf-8",
    )

    validate_links(tmp_path, "wiki")


def test_projected_html_heading_ids_are_validated_for_wiki(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        '<a id="stable-id"></a>\n# Descriptive heading\n\n[heading](#stable-id)\n',
        encoding="utf-8",
    )

    validate_links(tmp_path, "wiki")


def test_reference_link_definitions_and_usages_are_validated(tmp_path: Path):
    (tmp_path / "Target.md").write_text("# Present\n", encoding="utf-8")
    (tmp_path / "Page.md").write_text(
        "[target][full] and [target again][].\n\n[full]: Target.md#present\n[target again]: Target.md\n",
        encoding="utf-8",
    )

    validate_links(tmp_path, "site")


def test_escaped_reference_labels_are_validated(tmp_path: Path):
    (tmp_path / "Target.md").write_text("# Present\n", encoding="utf-8")
    (tmp_path / "Page.md").write_text(
        "[target][a\\]b]\n\n[a\\]b]: Target.md#present\n",
        encoding="utf-8",
    )

    validate_links(tmp_path, "site")


def test_undefined_reference_link_usage_is_rejected(tmp_path: Path):
    (tmp_path / "Page.md").write_text("[missing][reference]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="undefined site reference"):
        validate_links(tmp_path, "site")


def test_broken_reference_link_definition_is_rejected(tmp_path: Path):
    (tmp_path / "Page.md").write_text("[missing][reference]\n\n[reference]: Missing.md\n", encoding="utf-8")

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


@pytest.mark.parametrize(
    "target",
    [
        "https://thekaveh.github.io/NNx/Concepts/",
        "https://github.com/thekaveh/NNx/wiki/Concepts",
        "https://github.com/thekaveh/NNx/blob/main/docs/concepts.md",
    ],
)
def test_generated_surfaces_reject_cross_surface_origins(tmp_path: Path, target: str):
    (tmp_path / "Page.md").write_text(f"[cross-surface]({target})\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cross-surface site link"):
        validate_links(tmp_path, "site")


@pytest.mark.parametrize("surface", ["site", "wiki"])
def test_generated_surfaces_reject_root_relative_links(tmp_path: Path, surface: str):
    (tmp_path / "Page.md").write_text("[ambiguous](/Concepts)\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root-relative"):
        validate_links(tmp_path, surface)


def test_css_comments_do_not_create_dependencies(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "theme.css").write_text("/* url(missing.png) */\nbody { color: black; }\n", encoding="utf-8")

    validate_links(tmp_path, "site")


def test_css_strings_do_not_create_dependencies(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "theme.css").write_text('body::before { content: "url(missing.png)"; }\n', encoding="utf-8")

    validate_links(tmp_path, "site")


def test_css_image_set_dependencies_are_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "theme.css").write_text('body { background: image-set("missing.png" 1x); }\n', encoding="utf-8")

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


def test_css_image_set_escaped_parenthesis_does_not_hide_dependencies(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "diagram").write_text("placeholder\n", encoding="utf-8")
    (tmp_path / "theme.css").write_text(
        r'body { background: image-set(url(diagram\).png) 1x, "missing.png" 2x); }' + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


@pytest.mark.parametrize(
    "css", [r"body { background: u\72l(missing.png); }", r"body { background: url(missing\2e png); }"]
)
def test_css_escaped_dependencies_are_validated(tmp_path: Path, css: str):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "theme.css").write_text(css + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


@pytest.mark.parametrize("escape", [r"\0", r"\110000", r"\D800"])
def test_css_invalid_escape_codepoints_use_replacement_character(tmp_path: Path, escape: str):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "�").write_text("asset\n", encoding="utf-8")
    (tmp_path / "theme.css").write_text(f"body {{ background: url({escape}); }}\n", encoding="utf-8")
    validate_links(tmp_path, "site")


def test_css_quoted_imports_are_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "theme.css").write_text('@import "missing.css";\n', encoding="utf-8")

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


def test_css_escaped_import_keyword_is_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "theme.css").write_text(r'@\69mport "missing.css";' + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


def test_css_escaped_newline_is_a_continuation(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "present.png").write_bytes(b"image")
    (tmp_path / "theme.css").write_text('body { background: url("present\\\n.png"); }\n', encoding="utf-8")
    validate_links(tmp_path, "site")


def test_css_nested_image_function_is_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "theme.css").write_text('body { background: image-set(image("missing.png") 1x); }\n', encoding="utf-8")
    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


def test_css_directional_image_function_is_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "theme.css").write_text('body { background: image(ltr "missing.png"); }\n', encoding="utf-8")
    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


@pytest.mark.parametrize(
    "html",
    [
        '<style>body { background: url("missing.png"); }</style>',
        '<div style="background: url(/assets/missing.png)"></div>',
    ],
)
def test_html_css_resources_are_validated(tmp_path: Path, html: str):
    (tmp_path / "Page.md").write_text(html + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="(?:broken site link|root-relative site link)"):
        validate_links(tmp_path, "site")


def test_copied_svg_resources_are_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "diagram.svg").write_text('<svg><image href="missing.png"/></svg>\n', encoding="utf-8")

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


def test_svg_stylesheet_processing_instruction_is_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "diagram.svg").write_text(
        '<?xml-stylesheet type="text/css" href="missing.css"?>\n<svg/>\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


@pytest.mark.parametrize(
    "wrapper", ['<!-- <?xml-stylesheet href="missing.css"?> -->', '<![CDATA[<?xml-stylesheet href="missing.css"?>]]>']
)
def test_inactive_svg_stylesheet_text_is_ignored(tmp_path: Path, wrapper: str):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "diagram.svg").write_text(f"<svg>{wrapper}</svg>\n", encoding="utf-8")
    validate_links(tmp_path, "site")


def test_svg_stylesheet_href_text_inside_title_is_ignored(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "diagram.svg").write_text("<?xml-stylesheet title=\"href='missing.css'\"?>\n<svg/>\n", encoding="utf-8")
    validate_links(tmp_path, "site")


def test_svg_stylesheet_duplicate_href_is_rejected(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "diagram.svg").write_text(
        '<?xml-stylesheet href="missing.css" href="present.css"?>\n<svg/>\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate processing-instruction attribute"):
        validate_links(tmp_path, "site")


@pytest.mark.parametrize(
    "element",
    [
        "feImage",
        "script",
        "mpath",
        "textPath",
        "filter",
        "pattern",
        "linearGradient",
        "radialGradient",
        "cursor",
        "animate",
        "animateMotion",
        "animateTransform",
        "set",
    ],
)
def test_additional_svg_resources_are_validated(tmp_path: Path, element: str):
    (tmp_path / "Page.md").write_text("# Page\n", encoding="utf-8")
    (tmp_path / "diagram.svg").write_text(
        f'<svg><{element} href="missing.bin"/></svg>\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


def test_wiki_anchor_for_literal_html_code_heading(tmp_path: Path):
    (tmp_path / "Page.md").write_text("# `<tag>`\n\n[tag](#tag)\n", encoding="utf-8")

    validate_links(tmp_path, "wiki")


def test_pyright_configuration_includes_documentation_scripts():
    config = build_docs.ROOT / "pyrightconfig.json"
    included = __import__("json").loads(config.read_text(encoding="utf-8"))["include"]

    assert "scripts/docs" in included


def test_diagram_extraction_balances_nested_svg(tmp_path: Path):
    from scripts.docs.extract_architecture_svg import render

    master = tmp_path / "diagram.html"
    master.write_text('<html><svg id="outer"><svg id="inner"></svg><text>after</text></svg></html>', encoding="utf-8")

    assert render(master) == '<svg id="outer"><svg id="inner"></svg><text>after</text></svg>\n'


def test_local_link_lookup_decodes_path_and_ignores_query(tmp_path: Path):
    (tmp_path / "Target copy(1).md").write_text("# Present\n", encoding="utf-8")
    (tmp_path / "Page.md").write_text(
        "[target](Target%20copy\\(1\\).md?view=full#present)\n",
        encoding="utf-8",
    )

    validate_links(tmp_path, "site")


def test_links_inside_multiline_html_comments_are_not_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        "<!--\n[inline](Missing.md)\n[reference][missing]\n-->\n",
        encoding="utf-8",
    )

    validate_links(tmp_path, "site")


def test_multiline_inline_links_are_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("[missing](\nMissing.md\n)\n", encoding="utf-8")

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


def test_multiline_reference_definitions_are_validated(tmp_path: Path):
    (tmp_path / "Target.md").write_text("# Present\n", encoding="utf-8")
    (tmp_path / "Page.md").write_text(
        '[target][ref]\n\n[ref]:\n    Target.md#present "Docs"\n',
        encoding="utf-8",
    )

    validate_links(tmp_path, "site")


def test_broken_multiline_reference_definitions_are_rejected(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        "[missing][ref]\n\n[ref]:\n    Missing.md\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


def test_links_inside_complete_inline_html_comments_are_not_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        "<!-- [inline](Missing.md) and [reference][missing] -->\n",
        encoding="utf-8",
    )

    validate_links(tmp_path, "site")


def test_link_after_complete_inline_html_comment_is_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        "<!-- [ignored](Also-Missing.md) --> [missing](Missing.md)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r": Missing\.md$"):
        validate_links(tmp_path, "site")


def test_html_comment_marker_inside_code_span_does_not_hide_later_links(tmp_path: Path):
    (tmp_path / "Page.md").write_text(
        "Use `<!--` literally.\n\n[missing](Missing.md)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="broken site link"):
        validate_links(tmp_path, "site")


def test_links_inside_four_space_indented_code_are_not_validated(tmp_path: Path):
    (tmp_path / "Page.md").write_text("    [example](Missing.md)\n", encoding="utf-8")

    validate_links(tmp_path, "site")


@pytest.mark.parametrize("surface", ["site", "wiki"])
@pytest.mark.parametrize(
    "heading",
    [
        "> ## Blockquoted heading",
        "> Blockquoted heading\n> -------------------",
    ],
)
def test_blockquoted_heading_anchors_are_validated(tmp_path: Path, surface: str, heading: str):
    (tmp_path / "Page.md").write_text(
        f"{heading}\n\n[heading](#blockquoted-heading)\n",
        encoding="utf-8",
    )

    validate_links(tmp_path, surface)
