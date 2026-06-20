from macorag.cli import build_parser, main


def test_cli_parser_has_required_commands():
    parser = build_parser()

    commands = parser._subparsers._group_actions[0].choices

    assert "build-canonical" in commands
    assert "build-linearrag" in commands
    assert "sample" in commands
    assert "generate-trajectories" in commands
    assert "filter-trajectories" in commands


def test_build_canonical_defaults():
    args = build_parser().parse_args(["build-canonical"])

    assert args.command == "build-canonical"
    assert args.data_root == "data"
    assert args.output_root == "data/processed"


def test_build_linearrag_defaults():
    args = build_parser().parse_args(["build-linearrag"])

    assert args.command == "build-linearrag"
    assert args.processed_root == "data/processed"
    assert args.output_root == "linearrag_dataset"


def test_sample_defaults():
    args = build_parser().parse_args(["sample"])

    assert args.command == "sample"
    assert args.processed_root == "data/processed"
    assert args.output_root == "trajectories"
    assert args.per_dataset == 1000
    assert args.seed == 7


def test_generate_trajectories_defaults():
    args = build_parser().parse_args(["generate-trajectories"])

    assert args.command == "generate-trajectories"
    assert args.linearrag_root == "linearrag_dataset"
    assert args.sample_root == "trajectories"
    assert args.raw_output_name == "raw_teacher_trajectories.jsonl"


def test_filter_trajectories_defaults():
    args = build_parser().parse_args(["filter-trajectories"])

    assert args.command == "filter-trajectories"
    assert args.trajectory_root == "trajectories"
    assert args.linearrag_root == "linearrag_dataset"
    assert args.filtered_output_name == "filtered_sft_trajectories.jsonl"


def test_main_parses_sample_and_returns_zero():
    assert main(["sample"]) == 0
