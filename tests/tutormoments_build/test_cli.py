"""Tests for the tutormoments-build CLI (tutormoments_build.cli).

Carved out of tests/tutormoments/test_cli.py when the build subcommands moved
out of the runtime CLI.
"""

from unittest.mock import patch


def test_tutormoments_build_dispatches(tmp_path):
    """main(['dataset', 'build', ...]) calls moments_build._cli_build with correct args."""
    from tutormoments_build.cli import main

    ids_file = tmp_path / "ids.json"
    ids_file.write_text("[]", encoding="utf-8")

    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    out_dir = tmp_path / "release"

    with patch("tutormoments_build.moments_build._cli_build") as mock_build:
        main(
            [
                "dataset",
                "build",
                "--set",
                "balanced_520",
                "--ids",
                str(ids_file),
                "--ground-truth",
                str(gt_dir),
                "--transcripts",
                str(tx_dir),
                "--out",
                str(out_dir),
                "--created",
                "2026-06-26",
            ]
        )

    mock_build.assert_called_once()
    call_args = mock_build.call_args[0][0]
    assert call_args.set == "balanced_520"
    assert call_args.ids == str(ids_file)
    assert call_args.ground_truth == str(gt_dir)
    assert call_args.transcripts == str(tx_dir)
    assert call_args.out == str(out_dir)
    assert call_args.created == "2026-06-26"


def test_dataset_validate_cli(capsys):
    """main(['dataset', 'validate', ...]) validates the mini fixture release dir."""
    from tutormoments_build.cli import main

    main(
        [
            "dataset",
            "validate",
            "--data_path",
            "tests/tutormoments/fixtures/mini_release",
        ]
    )

    captured = capsys.readouterr()
    assert "Dataset valid: mini_set" in captured.out


def test_dataset_build_writes_build_log(tmp_path):
    """dataset build drops a build.log (with the invoked command) in --out."""
    from tutormoments_build.cli import main

    ids_file = tmp_path / "ids.json"
    ids_file.write_text("[]", encoding="utf-8")
    out_dir = tmp_path / "release"

    with patch("tutormoments_build.moments_build._cli_build"):
        main(
            [
                "dataset",
                "build",
                "--set",
                "balanced_520",
                "--ids",
                str(ids_file),
                "--ground-truth",
                str(tmp_path),
                "--transcripts",
                str(tmp_path),
                "--out",
                str(out_dir),
            ]
        )

    content = (out_dir / "build.log").read_text(encoding="utf-8")
    assert "Command: tutormoments-build dataset build" in content


def test_build_ground_truth_dry_run_writes_nothing(tmp_path):
    """--dry-run must not create the out dir or a build.log."""
    from tutormoments_build.cli import main

    ann = tmp_path / "ann.jsonl"
    ann.write_text("", encoding="utf-8")
    out_dir = tmp_path / "gt"

    with patch("tutormoments_build.groundtruth.build_ground_truth"):
        main(
            [
                "dataset",
                "build-ground-truth",
                "--input",
                str(ann),
                "--out-dir",
                str(out_dir),
                "--dry-run",
            ]
        )

    assert not out_dir.exists()


def test_build_cli_accepts_log_flags(tmp_path):
    """--log-file after a dataset subcommand configures file logging."""
    import logging

    from tutormoments_build.cli import main

    log_file = tmp_path / "build.log"
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    try:
        main(
            [
                "dataset",
                "validate",
                "--data_path",
                "tests/tutormoments/fixtures/mini_release",
                "--log-file",
                str(log_file),
            ]
        )
        for handler in root.handlers:
            handler.flush()
    finally:
        for handler in root.handlers[:]:
            if handler not in old_handlers:
                root.removeHandler(handler)
                handler.close()

    content = log_file.read_text(encoding="utf-8")
    assert "Command: tutormoments-build dataset validate" in content


def test_tutormoments_build_ground_truth_help_parses():
    import pytest

    from tutormoments_build import cli as cli_mod

    parser = cli_mod._build_parser()
    # --help raises SystemExit(0) after printing; just confirm the path parses.
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["dataset", "build-ground-truth", "--help"])
    assert exc.value.code == 0


def test_tutormoments_build_ground_truth_dispatches_with_defaults(tmp_path):
    from tutormoments_build import cli as cli_mod

    ann = tmp_path / "ann.jsonl"
    ann.write_text("", encoding="utf-8")
    out = tmp_path / "gt"
    captured = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        return {"dry_run": kwargs.get("dry_run")}

    with patch(
        "tutormoments_build.groundtruth.build_ground_truth", side_effect=_fake_build
    ) as m:
        cli_mod.main(
            [
                "dataset",
                "build-ground-truth",
                "--input",
                str(ann),
                "--out-dir",
                str(out),
                "--dry-run",
            ]
        )
    assert m.called
    assert captured["dry_run"] is True
    assert str(captured["input_path"]) == str(ann)
    assert str(captured["out_dir"]) == str(out)
    assert captured["labeller"] == "hybrid"


def test_tutormoments_build_ground_truth_default_out_dir_follows_labeller(tmp_path):
    from tutormoments_build import cli as cli_mod

    ann = tmp_path / "ann.jsonl"
    ann.write_text("", encoding="utf-8")
    captured = {}
    with patch(
        "tutormoments_build.groundtruth.build_ground_truth",
        side_effect=lambda **kw: captured.update(kw),
    ):
        cli_mod.main(
            [
                "dataset",
                "build-ground-truth",
                "--input",
                str(ann),
                "--labeller",
                "hybrid",
                "--dry-run",
            ]
        )
    assert str(captured["out_dir"]).endswith("ground_truth_hybrid")
