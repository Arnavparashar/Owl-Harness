import json

from owl.datasets import Dataset, FieldMap, csv_dataset, hash_samples, jsonl_dataset
from owl.types import Sample


def test_jsonl_loader_maps_fields(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"text": "a", "label": 1, "src": "x"}) + "\n\n" + json.dumps({"text": "b", "label": 0, "src": "y"}) + "\n",
        encoding="utf-8",
    )
    ds = jsonl_dataset(p, FieldMap(input="text", target="label", metadata=["src"]))
    assert [(s.id, s.input, s.target, s.metadata) for s in ds.samples] == [
        ("0", "a", "1", {"src": "x"}),
        ("1", "b", "0", {"src": "y"}),
    ]


def test_csv_loader_uses_id_column(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("uid,q,a\nr1,hello,1\nr2,world,0\n", encoding="utf-8")
    ds = csv_dataset(p, FieldMap(input="q", target="a", id="uid"))
    assert [s.id for s in ds.samples] == ["r1", "r2"]
    assert ds.samples[0].target == "1"


def test_content_hash_is_stable_and_sensitive():
    a = [Sample(id="0", input="x", target="1")]
    b = [Sample(id="0", input="x", target="1")]
    c = [Sample(id="0", input="x", target="0")]
    assert hash_samples(a) == hash_samples(b)
    assert hash_samples(a) != hash_samples(c)
    assert Dataset(name="d", samples=a).content_hash == hash_samples(a)


def test_content_hash_depends_on_order():
    s1 = Sample(id="0", input="x", target="1")
    s2 = Sample(id="1", input="y", target="0")
    assert hash_samples([s1, s2]) != hash_samples([s2, s1])
