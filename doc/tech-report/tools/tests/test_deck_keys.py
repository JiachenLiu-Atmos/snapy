"""Tests of tools/gen_deck_keys.py, the Appendix I generator, on small parsers written like snapy's.

Run from the report directory: `python3 -m pytest tools/tests`.
"""

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))

import gen_deck_keys as g  # noqa: E402

PARSER = r'''
FooOptions FooOptionsImpl::from_yaml(std::string const& filename,
                                     bool verbose) {
  auto config = YAML::LoadFile(filename);
  auto node = config["foo"];
  check_keys(node, "foo", {"alpha", "name", "flag", "must", "sub", "table", "unused"},
             "pyharp", {"cfl"});
  op->alpha() = node["alpha"].as<double>(1.e-6);
  op->verbose() = node["name"].as<bool>(verbose);
  if (node["flag"]) {
    auto const text = node["flag"].Scalar();
    op->flag() = text == "true";
  }
  TORCH_CHECK(node["must"], "FooOptions: must is required (no default).");
  op->must() = node["must"].as<double>();
  auto take = [&](char const* key, double fallback) {
    if (!node[key]) return fallback;
    return node[key].as<double>();
  };
  op->beta() = take("table", 2.5);
  check_keys(node["sub"], "foo/sub", {"type"});
  op->sub() = BarOptionsImpl::from_yaml(node["sub"]);
  return op;
}
'''

READER = r'''
BarOptions BarOptionsImpl::from_yaml(YAML::Node const& node) {
  auto op = BarOptionsImpl::create();
  op->type() = node["type"].as<std::string>("roe");
  return op;
}
'''

HEADER = "struct FooOptionsImpl {\n  ADD_ARG(bool, flag) = false;\n};\n"


def tree(tmp_path):
    (tmp_path / "src" / "foo").mkdir(parents=True)
    (tmp_path / "src" / "foo" / "foo.cpp").write_text(PARSER)
    (tmp_path / "src" / "foo" / "bar.cpp").write_text(READER)
    (tmp_path / "src" / "foo" / "foo.hpp").write_text(HEADER)
    return tmp_path


def test_every_listed_key_once_with_its_default(tmp_path):
    rows = {f"{r['path']}/{r['key']}": r for r in g.extract(tree(tmp_path))}
    assert sorted(rows) == ["foo/alpha", "foo/cfl", "foo/flag", "foo/must", "foo/name", "foo/sub",
                            "foo/sub/type", "foo/table", "foo/unused"]
    d = {k: (r["default"]["kind"], r["default"]["value"]) for k, r in rows.items()}
    assert d["foo/alpha"] == ("fallback", "1.e-6")
    assert d["foo/name"] == ("fallback", "verbose")
    assert d["foo/flag"] == ("struct", "false")          # guarded read; the options struct's value stands
    assert d["foo/must"] == ("required", "")
    assert d["foo/table"] == ("fallback", "2.5")         # helper called with the key and a fallback
    assert d["foo/sub"] == ("block", "")
    assert d["foo/sub/type"] == ("fallback", '"roe"')    # block handed whole to another reader
    assert d["foo/unused"] == ("unread", "")
    assert d["foo/cfl"] == ("library", "")
    assert rows["foo/cfl"]["reader"] == "pyharp"


def test_a_read_of_the_same_key_in_another_block_is_not_its_default(tmp_path):
    t = tree(tmp_path)
    (t / "src" / "foo" / "foo.cpp").write_text(PARSER.replace('op->alpha() = node["alpha"].as<double>(1.e-6);',
                                                              'op->alpha() = other["alpha"].as<double>(9.);'))
    rows = {f"{r['path']}/{r['key']}": r for r in g.extract(t)}
    assert rows["foo/alpha"]["default"]["kind"] != "fallback"


def test_section_link_needs_the_path_or_the_enclosing_block():
    row = {"path": "forcing/top-cool", "key": "tau"}
    secs = [("sec-a", "A", ["tau"]),                          # the key alone: another block's tau
            ("sec-b", "B", ["top-cool", "tau", "tau"]),       # key with its block
            ("sec-c", "C", ["forcing/top-cool/tau"])]         # the full path
    assert g.explaining(row, secs) == ("sec-c", "C")
    assert g.explaining(row, secs[:2]) == ("sec-b", "B")
    assert g.explaining(row, secs[:1]) is None


def test_yaml_example_paths():
    assert g._yaml_paths(["forcing:", "  const-gravity:", "    grav1: -9.8  # m s^-2"]) == [
        "forcing", "forcing/const-gravity", "forcing/const-gravity/grav1"]


def test_committed_fragment_is_current():
    assert g.main(["--check"]) == 0


def test_anchor_is_on_the_cited_line():
    """The anchor a link names is an identifier of the cited line (STYLE 3.1): the key, its `_` spelling, or a call."""
    assert g._anchor('  op->alpha() = node["alpha"].as<double>(1.e-6);', "alpha") == "alpha"
    assert g._anchor("  ADD_ARG(bool, at_face) = false;", "at-face") == "at_face"
    assert g._anchor('  check_keys(node, "foo",', "flag") == "check_keys"
    assert g._anchor("", "flag") is None


def test_every_read_at_link_of_the_committed_fragment_names_its_anchor():
    """check_citations needs an anchor symbol for each line link; the committed App. I gives every one."""
    import check_citations as cc
    links = [c for c in cc.find_citations(g.FRAGMENT) if c.form == "url" and c.ranges]
    assert links and all(c.anchor for c in links), [c.text for c in links if not c.anchor][:3]


# --- the options schema (src/options/*.yaml) and the parsers' struct defaults -----------------------------------

GEN_OPTIONS = r'''
import yaml
from pathlib import Path


def row(d):
    r = dict(read="fallback" if d.get("key") else "none", alias={})
    r.update(d)
    return r


def load_tables(paths):
    out = []
    for p in paths:
        for d in yaml.safe_load_all(Path(p).read_text()):
            d = dict(dict(extra_keys={}, checks=[]), **d)
            d["rows"] = [row(r) for r in d["options"]]
            out.append(d)
    return out


def lit(r, v):
    if isinstance(v, dict) and "cpp" in v:
        return v["cpp"]
    if v is None:
        return None
    if r["type"] == "string":
        return '"' + str(v) + '"'
    return {"bool": lambda: "true" if v else "false", "int": lambda: str(int(v)),
            "double": lambda: repr(float(v))}[r["type"]]()
'''

SCHEMA = r'''module: bar
yaml_node: foo/bar
extra_keys:
  "": [inner, dead]
options:
  - {name: speed, key: speed, type: double, default: 2., doc: "a fallback"}
  - {name: mode, key: mode, type: string, default: slow, read: if_present, doc: "kept by the struct"}
  - {name: list, key: list, type: double, read: if_present, doc: "no default"}
  - {name: lo, key: inner/lo, alias: {lo_pi: pi}, type: double, default: 0., doc: "a sub-block row"}
  - {name: need, key: need, type: double, default: 1., doc: "required by a check"}
checks:
  - {cond: 'node["need"]', message: "need is required"}
'''

STRUCT_PARSER = r'''
BazOptions BazOptionsImpl::from_yaml(YAML::Node const& node) {
  auto op = BazOptionsImpl::create();
  check_keys(node, "baz", {"gamma", "type"});
  op->gamma() = node["gamma"].as<double>(op->gamma());
  op->type() = node["type"].as<std::string>(op->type());
  return op;
}
'''


def schema_tree(tmp_path):
    t = tree(tmp_path)
    (t / "tools").mkdir()
    (t / "tools" / "gen_options.py").write_text(GEN_OPTIONS)
    (t / "src" / "options").mkdir()
    (t / "src" / "options" / "bar.yaml").write_text(SCHEMA)
    (t / "src" / "baz").mkdir()
    (t / "src" / "baz" / "baz.cpp").write_text(STRUCT_PARSER)
    (t / "src" / "baz" / "baz.hpp").write_text(
        'struct BazOptionsImpl {\n  ADD_ARG(double, gamma) = 1.4;\n  ADD_ARG(std::string, type) = "hllc";\n};\n')
    (t / "src" / "foo" / "other.hpp").write_text('struct X {\n  ADD_ARG(std::string, type) = "roe";\n};\n')
    sys.modules.pop("gen_options", None)
    return t


def test_the_schema_rows_are_keys_with_their_defaults(tmp_path):
    rows = {f"{r['path']}/{r['key']}": r for r in g.extract(schema_tree(tmp_path))}
    d = {k: (r["default"]["kind"], r["default"]["value"]) for k, r in rows.items() if k.startswith("foo/bar")}
    assert d["foo/bar/speed"] == ("fallback", "2.0")
    assert d["foo/bar/mode"] == ("struct", '"slow"')
    assert d["foo/bar/list"] == ("guarded", "")
    assert d["foo/bar/inner/lo"] == ("fallback", "0.0")
    assert d["foo/bar/inner/lo_pi"] == ("guarded", "")
    assert d["foo/bar/need"] == ("required", "")
    assert d["foo/bar/inner"] == ("block", "")
    assert d["foo/bar/dead"] == ("unread", "")
    cited = rows["foo/bar/dead"]["default"]
    line = (tmp_path / cited["file"]).read_text().split("\n")[cited["line"] - 1]
    assert "dead" in line  # the extra_keys line that names the key


def test_a_struct_fallback_is_the_options_default_of_the_parsers_own_struct(tmp_path):
    rows = {f"{r['path']}/{r['key']}": r for r in g.extract(schema_tree(tmp_path))}
    assert (rows["baz/gamma"]["default"]["kind"], rows["baz/gamma"]["default"]["value"]) == ("struct", "1.4")
    # `type` is a member of two structs; the one beside the parser wins
    assert rows["baz/type"]["default"]["value"] == '"hllc"'
    assert rows["baz/type"]["default"]["file"] == "src/baz/baz.hpp"
