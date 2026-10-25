import json

import pytest

from jarvis.voice.wer import missed_words, score_archive, wer


def test_wer_identical_zero():
    assert wer("turn on the fan", "turn on the fan") == 0.0


def test_wer_one_sub():
    assert wer("turn on the fan", "turn on the van") == pytest.approx(0.25)


def test_wer_ignores_case_punct():
    assert wer("Turn on, the fan!", "turn on the FAN") == 0.0


def test_wer_empty_hyp_is_all_deletions():
    assert wer("turn on", "") == 1.0


def test_missed_words_counts_reference_words_not_matched():
    assert missed_words("open the vault now", "open a vault") == ["the", "now"]


def test_score_archive_uses_only_corrected_rows(tmp_path):
    rows = [
        {"ts": "1", "wav": "a.wav", "transcript": "turn on the van", "corrected": "turn on the fan"},
        {"ts": "2", "wav": "b.wav", "transcript": "hello", "corrected": None},
        {"ts": "3", "wav": "c.wav", "transcript": None, "corrected": "good night"},
    ]
    (tmp_path / "utterances.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result = score_archive(tmp_path)
    assert result.count == 2
    # 1 sub of 4 words + 2 deletions of 2 words = 3 errors / 6 words
    assert result.wer == pytest.approx(0.5)
    assert ("fan", 1) in result.top_missed


def test_score_archive_skips_malformed_lines(tmp_path):
    good = json.dumps({"ts": "1", "wav": "a", "transcript": "hi there", "corrected": "hi there"})
    (tmp_path / "utterances.jsonl").write_text(good + "\n{not json\n[1]\n")
    result = score_archive(tmp_path)
    assert result.count == 1
    assert result.skipped == 2


def test_cli_no_corrected_rows(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("voice_wer", Path("scripts/voice_wer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "utterances.jsonl").write_text(
        json.dumps({"ts": "1", "wav": "a", "transcript": "x", "corrected": None}) + "\n"
    )
    assert mod.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "no corrected rows yet" in out and "WER" not in out
