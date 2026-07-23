from pathlib import Path

import pytest

import scripts.docs.build_wiki as build_wiki
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


def test_rewrite_markdown_rejects_image_traversal_outside_repository(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    source = repository / "README.md"
    source.write_text("", encoding="utf-8")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")
    monkeypatch.setattr(build_wiki, "ROOT", repository)

    with pytest.raises(ValueError, match="outside the repository"):
        rewrite_markdown(source, "![Outside](../outside.png)\n", {})


def test_rewrite_markdown_rejects_repository_images_not_published_to_wiki(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    source = repository / "README.md"
    source.write_text("", encoding="utf-8")
    image = repository / "other" / "diagram.svg"
    image.parent.mkdir()
    image.write_text("<svg/>", encoding="utf-8")
    monkeypatch.setattr(build_wiki, "ROOT", repository)

    with pytest.raises(ValueError, match="not published"):
        rewrite_markdown(source, "![Diagram](other/diagram.svg)\n", {})


def test_rewrite_markdown_preserves_local_image_query_fragment_and_title(tmp_path: Path):
    source = tmp_path / "README.md"
    image = tmp_path / "diagram final.svg"
    source.write_text("", encoding="utf-8")
    image.write_text("<svg/>", encoding="utf-8")

    rendered = rewrite_markdown(
        source,
        '![Diagram](diagram%20final.svg?raw=1#layer "Architecture")\n',
        {},
    )

    assert rendered == '![Diagram](images/diagram%20final.svg?raw=1#layer "Architecture")\n'


def test_rewrite_markdown_maps_nested_labels_and_parenthesized_targets(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "guide(v(2)).md"
    source.write_text("", encoding="utf-8")
    target.write_text("# Guide\n", encoding="utf-8")

    rendered = rewrite_markdown(
        source,
        "[Guide [advanced]](guide(v(2)).md)\n",
        {target.resolve(): "Guide"},
    )

    assert rendered == "[Guide [advanced]](Guide)\n"


def test_rewrite_markdown_maps_angle_destination_with_parentheses(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "guide)draft(.md"
    source.write_text("", encoding="utf-8")
    target.write_text("# Guide\n", encoding="utf-8")

    rendered = rewrite_markdown(
        source,
        "[Guide](<guide)draft(.md>)\n",
        {target.resolve(): "Guide"},
    )

    assert rendered == "[Guide](Guide)\n"


def test_rewrite_markdown_preserves_escaped_opening_bracket(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "\\[Example](missing.md)\n"

    assert rewrite_markdown(source, text, {}) == text


def test_rewrite_markdown_rejects_missing_repo_links(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, "[Missing](missing.md)\n", {})


def test_rewrite_markdown_keeps_external_links(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    rendered = rewrite_markdown(source, "[PyTorch](https://pytorch.org)\n", {})
    assert "[PyTorch](https://pytorch.org)" in rendered


@pytest.mark.parametrize("target", ["//cdn.example.com/theme.css"])
def test_rewrite_markdown_preserves_absolute_web_paths(tmp_path: Path, target: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = f"[External]({target})\n"

    assert rewrite_markdown(source, text, {}) == text


def test_rewrite_markdown_maps_reference_link_definitions(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "api.md"
    source.write_text("", encoding="utf-8")
    target.write_text("# API\n", encoding="utf-8")
    text = '[API][reference] and [API again][].\n\n[reference]: api.md#types "API docs"\n[api again]: api.md\n'

    rendered = rewrite_markdown(source, text, {target.resolve(): "API-Reference"})

    assert rendered == (
        "[API][reference] and [API again][].\n\n"
        '[reference]: API-Reference#types "API docs"\n'
        "[api again]: API-Reference\n"
    )


def test_rewrite_markdown_maps_escaped_reference_labels(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "api.md"
    source.write_text("", encoding="utf-8")
    target.write_text("# API\n", encoding="utf-8")
    text = "[API][a\\]b]\n\n[a\\]b]: api.md\n"

    rendered = rewrite_markdown(source, text, {target.resolve(): "API-Reference"})

    assert rendered == "[API][a\\]b]\n\n[a\\]b]: API-Reference\n"


def test_rewrite_markdown_preserves_query_fragment_title_and_decodes_lookup_path(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "target copy(1).md"
    source.write_text("", encoding="utf-8")
    target.write_text("# Present\n", encoding="utf-8")

    rendered = rewrite_markdown(
        source,
        '[Target](target%20copy\\(1\\).md?view=full#present "Docs")\n',
        {target.resolve(): "Target"},
    )

    assert rendered == '[Target](Target?view=full#present "Docs")\n'


def test_rewrite_markdown_rejects_missing_reference_link_targets(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, "[Missing][ref]\n\n[ref]: missing.md\n", {})


def test_rewrite_markdown_projects_explicit_heading_ids_for_github_wiki(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "# ATX heading {#atx-id}\n\nSetext heading {#setext-id}\n---\n"

    rendered = rewrite_markdown(source, text, {}, surface="wiki")

    assert rendered == ('<a id="atx-id"></a>\n# ATX heading\n\n<a id="setext-id"></a>\nSetext heading\n---\n')


def test_rewrite_markdown_projects_combined_heading_attributes_for_github_wiki(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = '# Heading {#stable .featured data-kind="guide"}\n'

    rendered = rewrite_markdown(source, text, {}, surface="wiki")

    assert rendered == '<a id="stable"></a>\n# Heading\n'


def test_rewrite_markdown_projects_heading_ids_around_inline_code(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "# Use `[Example](missing.md)` safely {#stable}\n"

    rendered = rewrite_markdown(source, text, {}, surface="wiki")

    assert rendered == '<a id="stable"></a>\n# Use `[Example](missing.md)` safely\n'


def test_wiki_determinism_check_resolves_internal_temporary_paths(tmp_path: Path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    paths = iter((alias / "first", alias / "second"))

    class TemporaryDirectory:
        def __init__(self):
            self.path = next(paths)

        def __enter__(self):
            return str(self.path)

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(build_wiki.tempfile, "TemporaryDirectory", TemporaryDirectory)

    build_wiki.check()


@pytest.mark.parametrize("delimiter", ["`", "``"])
def test_rewrite_markdown_preserves_links_inside_inline_code(tmp_path: Path, delimiter: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = f"Use {delimiter}[Example](missing.md){delimiter} literally.\n"

    assert rewrite_markdown(source, text, {}) == text


@pytest.mark.parametrize("title", ['"API docs"', "'API docs'", "(API docs)"])
def test_rewrite_markdown_preserves_link_titles(tmp_path: Path, title: str):
    source = tmp_path / "README.md"
    target = tmp_path / "api.md"
    source.write_text("", encoding="utf-8")
    target.write_text("# API\n", encoding="utf-8")

    rendered = rewrite_markdown(source, f"[API](api.md {title})\n", {target.resolve(): "API-Reference"})

    assert rendered == f"[API](API-Reference {title})\n"


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_rewrite_markdown_ignores_links_inside_fences(tmp_path: Path, fence: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = f"{fence}markdown\n[Example](missing.md)\n{fence}\n"

    assert rewrite_markdown(source, text, {}) == text


def test_rewrite_markdown_ignores_links_inside_blockquoted_fences(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "> ~~~markdown\n> [Example](missing.md)\n> [ref]: missing.md\n> ~~~\n"

    assert rewrite_markdown(source, text, {}) == text


def test_rewrite_markdown_ignores_links_inside_pre_blocks(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "<pre>\n[Example](missing.md)\n[ref]: missing.md\n</pre>\n"

    assert rewrite_markdown(source, text, {}) == text


@pytest.mark.parametrize("opening,closing", [("~~~", "~~~"), ("<pre>", "</pre>")])
def test_blockquoted_suppression_ends_at_container_boundary(tmp_path: Path, opening: str, closing: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = f"> {opening}\n> [Literal](missing.md)\n[Real](missing.md)\n> {closing}\n"

    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, text, {})


def test_rewrite_markdown_ignores_blockquoted_indented_code(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = ">\n>     [Literal](missing.md)\n"

    assert rewrite_markdown(source, text, {}) == text


def test_rewrite_markdown_maps_blockquoted_reference_definitions(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "Target.md"
    source.write_text("", encoding="utf-8")
    target.write_text("# Target\n", encoding="utf-8")
    text = "> [Target][ref]\n>\n> [ref]: Target.md\n"

    assert rewrite_markdown(source, text, {target.resolve(): "Target"}) == "> [Target][ref]\n>\n> [ref]: Target\n"


def test_rewrite_markdown_maps_raw_html_links_and_images(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "Target.md"
    image = tmp_path / "diagram.svg"
    source.write_text("", encoding="utf-8")
    target.write_text("# Target\n", encoding="utf-8")
    image.write_text("<svg/>", encoding="utf-8")
    text = '<a href="Target.md">Target</a> <img src="diagram.svg" alt="Diagram">\n'

    assert rewrite_markdown(source, text, {target.resolve(): "Target"}) == (
        '<a href="Target">Target</a> <img src="images/diagram.svg" alt="Diagram">\n'
    )


@pytest.mark.parametrize(
    "text",
    [
        '<div><a href="Target.md">Target</a></div>\n',
        "<a href=Target.md>Target</a>\n",
    ],
)
def test_rewrite_markdown_maps_html_links_in_containers_and_unquoted_attributes(tmp_path: Path, text: str):
    source = tmp_path / "README.md"
    target = tmp_path / "Target.md"
    source.write_text("", encoding="utf-8")
    target.write_text("# Target\n", encoding="utf-8")

    rendered = rewrite_markdown(source, text, {target.resolve(): "Target"})
    assert "Target.md" not in rendered
    assert "href" in rendered and "Target" in rendered


def test_rewrite_markdown_ignores_markdown_like_html_attribute_text(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = '<table data-example="[Literal](missing.md)"><tr><td>ok</td></tr></table>\n'

    assert rewrite_markdown(source, text, {}) == text


def test_rewrite_markdown_ignores_links_inside_list_contained_fence(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "- ~~~markdown\n  [Literal](missing.md)\n  ~~~\n"

    assert rewrite_markdown(source, text, {}) == text


def test_rewrite_markdown_preserves_nested_asset_path(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repo"
    source = repository / "README.md"
    image = repository / "docs" / "assets" / "nested" / "icon.svg"
    source.parent.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    image.write_text("<svg/>", encoding="utf-8")
    monkeypatch.setattr(build_wiki, "ROOT", repository)

    assert rewrite_markdown(source, "![Icon](docs/assets/nested/icon.svg)\n", {}) == (
        "![Icon](images/nested/icon.svg)\n"
    )


@pytest.mark.parametrize(
    "text",
    [
        '```html\n<a href="missing.md">literal</a>\n```\n',
        '<!-- <a href="missing.md">literal</a> -->\n',
        '<pre><a href="missing.md">literal</a></pre>\n',
    ],
)
def test_rewrite_markdown_does_not_process_html_targets_in_literal_blocks(tmp_path: Path, text: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    assert rewrite_markdown(source, text, {}) == text


@pytest.mark.parametrize(
    "text",
    [
        '<a data-href="missing.md">literal</a>\n',
        '<div data-example="href=missing.md">literal</div>\n',
        '<div data-example="a > b">[Literal](missing.md)</div>\n',
    ],
)
def test_html_attribute_scanning_respects_exact_names_and_quoted_delimiters(tmp_path: Path, text: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    assert rewrite_markdown(source, text, {}) == text


def test_site_nested_asset_uses_assets_prefix(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repo"
    source = repository / "README.md"
    image = repository / "docs" / "assets" / "nested" / "icon.svg"
    source.parent.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    image.write_text("<svg/>", encoding="utf-8")
    monkeypatch.setattr(build_wiki, "ROOT", repository)

    assert rewrite_markdown(source, "![Icon](docs/assets/nested/icon.svg)\n", {}, surface="site") == (
        "![Icon](assets/nested/icon.svg)\n"
    )


def test_list_continuation_fence_is_suppressed(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "- item\n    ~~~markdown\n    [Literal](missing.md)\n    ~~~\n"
    assert rewrite_markdown(source, text, {}) == text


@pytest.mark.parametrize("tag", ["section", "article", "nav"])
def test_additional_raw_html_blocks_suppress_markdown(tmp_path: Path, tag: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = f"<{tag}>\n[Literal](missing.md)\n</{tag}>\n"
    assert rewrite_markdown(source, text, {}) == text


def test_nested_raw_html_blocks_remain_suppressed_to_outer_close(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "<div>\n<div>inner</div>\n[Literal](missing.md)\n</div>\n"
    assert rewrite_markdown(source, text, {}) == text


def test_html_entities_are_decoded_for_target_lookup(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "Target.md"
    source.write_text("", encoding="utf-8")
    target.write_text("# Target\n", encoding="utf-8")
    assert rewrite_markdown(source, '<a href="Target&#46;md">Target</a>\n', {target.resolve(): "Target"}) == (
        '<a href="Target">Target</a>\n'
    )


def test_duplicate_html_target_attributes_are_rejected(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate HTML attribute"):
        rewrite_markdown(source, '<a href="one.md" href="two.md">bad</a>\n', {})


def test_html_resource_attributes_are_rewritten(tmp_path: Path):
    source = tmp_path / "README.md"
    (tmp_path / "one.png").write_bytes(b"one")
    (tmp_path / "two.png").write_bytes(b"two")
    source.write_text("", encoding="utf-8")
    text = '<source src="one.png" srcset="one.png 1x, two.png 2x">\n'
    assert rewrite_markdown(source, text, {}) == (
        '<source src="images/one.png" srcset="images/one.png 1x, images/two.png 2x">\n'
    )


def test_list_continuation_linked_prose_is_validated(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, "- item\n    [Missing](missing.md)\n", {})


@pytest.mark.parametrize(
    "text",
    [
        "<textarea>\n[Literal](missing.md)\n</textarea>\n",
        "<![CDATA[\n[Literal](missing.md)\n]]>\n",
        "<?processing\n[Literal](missing.md)\n?>\n",
        "<!DECLARATION\n[Literal](missing.md)\n>\n",
    ],
)
def test_additional_literal_html_forms_suppress_markdown(tmp_path: Path, text: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    assert rewrite_markdown(source, text, {}) == text


@pytest.mark.parametrize(
    "html",
    [
        '<script src="missing.js"></script>',
        '<iframe src="missing.html"></iframe>',
        '<object data="missing.bin"></object>',
        '<embed src="missing.bin">',
        '<link href="missing.css">',
    ],
)
def test_standard_html_resources_are_validated(tmp_path: Path, html: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, html + "\n", {})


def test_srcset_data_url_is_preserved(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = '<img srcset="data:image/png;base64,AAAA 1x">\n'
    assert rewrite_markdown(source, text, {}) == text


def test_srcset_descriptorless_data_url_does_not_consume_local_candidate(tmp_path: Path):
    source = tmp_path / "README.md"
    image = tmp_path / "fallback.png"
    source.write_text("", encoding="utf-8")
    image.write_bytes(b"png")
    text = '<img srcset="data:image/png;base64,AAAA, fallback.png 2x">\n'

    assert rewrite_markdown(source, text, {}) == ('<img srcset="data:image/png;base64,AAAA, images/fallback.png 2x">\n')


def test_top_level_raw_html_block_ends_at_blank_line(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, "<div>\ncontent\n\n[Missing](missing.md)\n", {})


def test_raw_html_block_remains_literal_until_blank_after_closing_tag(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "<div>\ncontent\n</div>\n[Literal](missing.md)\n\n"

    assert rewrite_markdown(source, text, {}) == text.rstrip() + "\n"


def test_nested_list_raw_html_block_suppresses_markdown(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "- item\n    - <div>\n      [Literal](missing.md)\n\n"

    assert rewrite_markdown(source, text, {}) == text.rstrip() + "\n"


def test_raw_html_does_not_cross_list_container_boundary(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, "- <div>\n[Missing](missing.md)\n", {})


def test_html_comment_does_not_cross_list_container_boundary(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, "- <!-- comment\n[Missing](missing.md)\n", {})


def test_lowercase_html_declaration_does_not_suppress_markdown(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, "<!foo\n[Missing](missing.md)\n", {})


@pytest.mark.parametrize(
    "opening,closing",
    [("<?processing", "?>"), ("<![CDATA[", "]]>"), ("<!DOCTYPE html", ">")],
)
def test_list_contained_literal_html_forms_are_suppressed(tmp_path: Path, opening: str, closing: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = f"- {opening}\n  [Literal](missing.md)\n  {closing}\n"

    assert rewrite_markdown(source, text, {}) == text


@pytest.mark.parametrize("opening", ["<?processing", "<![CDATA[", "<!DOCTYPE html"])
def test_list_literal_html_does_not_cross_container_boundary(tmp_path: Path, opening: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, f"- {opening}\n[Missing](missing.md)\n", {})


def test_tab_indented_list_literal_html_remains_suppressed(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "- <?processing\n\t[Literal](missing.md)\n\t?>\n"
    assert rewrite_markdown(source, text, {}) == text


def test_tab_indented_list_fence_remains_suppressed(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "- ```markdown\n\t[Literal](missing.md)\n\t```\n"
    assert rewrite_markdown(source, text, {}) == text


def test_tab_indented_list_reference_definition_is_projected(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "Target.md"
    source.write_text("", encoding="utf-8")
    target.write_text("# Target\n", encoding="utf-8")
    rendered = rewrite_markdown(source, "- [Target][ref]\n\t[ref]: Target.md\n", {target.resolve(): "Target"})
    assert "[ref]: Target" in rendered


def test_partial_tab_after_list_indent_remains_prose(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, "- item\n\n  \t[Missing](Missing.md)\n", {})


def test_dedented_closer_ends_list_contained_fence(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "- ```markdown\n  [Literal](missing.md)\n```\n[Missing](missing.md)\n"
    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, text, {})


def test_nested_list_setext_heading_id_is_projected(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "- item\n    - Nested heading {#stable}\n      -----------------------\n"

    rendered = rewrite_markdown(source, text, {}, surface="wiki")

    assert "{#stable}" not in rendered
    assert '<a id="stable"></a>' in rendered


def test_rewrite_markdown_projects_nested_list_heading_ids(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "- item\n    - ## Nested heading {#stable}\n"

    rendered = rewrite_markdown(source, text, {}, surface="wiki")

    assert rendered == '- item\n<a id="stable"></a>\n    - ## Nested heading\n'


@pytest.mark.parametrize("target", ["/Concepts", "/assets/diagram.svg"])
def test_rewrite_markdown_rejects_root_relative_targets(tmp_path: Path, target: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="root-relative"):
        rewrite_markdown(source, f"[Target]({target})\n", {})


@pytest.mark.parametrize("tag", ["script", "style", "div"])
def test_rewrite_markdown_ignores_markdown_inside_raw_html_blocks(tmp_path: Path, tag: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = f"<{tag}>\n[Literal](missing.md)\n</{tag}>\n"

    assert rewrite_markdown(source, text, {}) == text


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_rewrite_markdown_does_not_close_fence_on_info_like_content(tmp_path: Path, fence: str):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = f"{fence}text\n{fence}python\n[Example](missing.md)\n{fence}\n"

    assert rewrite_markdown(source, text, {}) == text


def test_rewrite_markdown_ignores_multiline_html_comments(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "<!--\n[Example](missing.md)\n[Example][missing]\n-->\n"

    assert rewrite_markdown(source, text, {}) == text


def test_rewrite_markdown_maps_multiline_inline_links(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "target copy.md"
    source.write_text("", encoding="utf-8")
    target.write_text("# Target\n", encoding="utf-8")
    text = '[Target](\ntarget%20copy.md?view=full#target "Docs"\n)\n'

    rendered = rewrite_markdown(source, text, {target.resolve(): "Target"})

    assert rendered == '[Target](Target?view=full#target "Docs")\n'


def test_rewrite_markdown_maps_multiline_reference_definitions(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "target.md"
    source.write_text("", encoding="utf-8")
    target.write_text("# Target\n", encoding="utf-8")
    text = '[Target][ref]\n\n[ref]:\n    target.md#target "Docs"\n'

    rendered = rewrite_markdown(source, text, {target.resolve(): "Target"})

    assert rendered == '[Target][ref]\n\n[ref]: Target#target "Docs"\n'


def test_rewrite_markdown_ignores_complete_inline_html_comments(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "<!-- [Example](missing.md) -->\n"

    assert rewrite_markdown(source, text, {}) == text


def test_rewrite_markdown_does_not_open_html_comment_inside_code_span(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="broken local documentation link"):
        rewrite_markdown(source, "Use `<!--` literally.\n\n[Example](missing.md)\n", {})


def test_rewrite_markdown_ignores_four_space_indented_code(tmp_path: Path):
    source = tmp_path / "README.md"
    source.write_text("", encoding="utf-8")
    text = "    [Example](missing.md)\n"

    assert rewrite_markdown(source, text, {}) == text


def test_rewrite_markdown_rejects_existing_unmapped_document(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "orphan.md"
    source.write_text("", encoding="utf-8")
    target.write_text("# Orphan\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unmapped local documentation link"):
        rewrite_markdown(source, "[Orphan](orphan.md)\n", {})


def test_rewrite_markdown_rejects_existing_unmapped_non_markdown_file(tmp_path: Path):
    source = tmp_path / "README.md"
    target = tmp_path / "pyproject.toml"
    source.write_text("", encoding="utf-8")
    target.write_text("[project]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unmapped local documentation link"):
        rewrite_markdown(source, "[Configuration](pyproject.toml)\n", {})
