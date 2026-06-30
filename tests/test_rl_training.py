from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import torch

from rag import AgentRole, RAGState
from rl_training.config import parse_args
from rl_training.data import load_rl_samples
from rl_training.policy import HFSharedPolicy
from rl_training.policy import sequence_logprobs
from rl_training.rewards import compute_answer_f1, compute_rl_rewards
from rl_training.train_grpo_macorag import _parse_gpu_indices
from rl_training.train_grpo_macorag import _build_policy
from rl_training.train_grpo_macorag import _train_on_rollouts
from rl_training.train_grpo_macorag import _validate_vllm_gpu_placement
from rl_training.train_grpo_macorag import _write_train_event
from rl_training.trainer import compute_grpo_loss
from rl_training.vllm_client import collect_trainable_named_parameters


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_parse_args_loads_train_grpo_yaml(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text(
        "\n".join(
            [
                'model_path: "model/base"',
                'sft_adapter_path: "outputs/sft/adapter"',
                'rl_data_root: "data/rl/train"',
                'retrieval_root: "data/trajectory_train_retrieval"',
                'output_dir: "outputs/grpo"',
                "max_samples: 8",
                "max_rounds: 2",
                "group_size: 4",
                "kl_beta: 0.03",
                "clip_epsilon: 0.15",
                "learning_rate: 0.00001",
                "per_device_train_batch_size: 1",
                "gradient_accumulation_steps: 2",
                "gpu_indices: \"0,1\"",
                "disable_tqdm: false",
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config), "--max-samples", "3"])

    assert args.model_path == "model/base"
    assert args.sft_adapter_path == "outputs/sft/adapter"
    assert args.rl_data_root == "data/rl/train"
    assert args.retrieval_root == "data/trajectory_train_retrieval"
    assert args.output_dir == "outputs/grpo"
    assert args.max_samples == 3
    assert args.max_rounds == 2
    assert args.group_size == 4
    assert args.kl_beta == 0.03
    assert args.clip_epsilon == 0.15
    assert args.gradient_accumulation_steps == 2
    assert args.gpu_indices == "0,1"
    assert args.disable_tqdm is False


def test_parse_args_supports_disabling_rl_progress_bar(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text("disable_tqdm: true\n", encoding="utf-8")

    args = parse_args(["--config", str(config)])

    assert args.disable_tqdm is True


def test_parse_args_loads_vllm_generation_config(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text(
        "\n".join(
            [
                "use_vllm_generation: true",
                'vllm_host: "127.0.0.1"',
                "vllm_port: 8123",
                'vllm_gpu_indices: "0"',
                "vllm_tensor_parallel_size: 1",
                "vllm_gpu_memory_utilization: 0.70",
                "vllm_max_model_len: 4608",
                'vllm_dtype: "auto"',
                "vllm_sync_after_step: true",
                "vllm_sync_trainable_only: true",
                "vllm_timeout_seconds: 90",
                'gpu_indices: "1"',
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.use_vllm_generation is True
    assert args.vllm_host == "127.0.0.1"
    assert args.vllm_port == 8123
    assert args.vllm_gpu_indices == "0"
    assert args.vllm_tensor_parallel_size == 1
    assert args.vllm_gpu_memory_utilization == 0.70
    assert args.vllm_max_model_len == 4608
    assert args.vllm_dtype == "auto"
    assert args.vllm_sync_after_step is True
    assert args.vllm_sync_trainable_only is True
    assert args.vllm_timeout_seconds == 90
    assert args.gpu_indices == "1"


def test_parse_gpu_indices_normalizes_comma_lists() -> None:
    assert _parse_gpu_indices("0, 1") == {"0", "1"}
    assert _parse_gpu_indices(2) == {"2"}
    assert _parse_gpu_indices("") == set()
    assert _parse_gpu_indices(None) == set()


def test_validate_vllm_gpu_placement_rejects_overlap() -> None:
    args = Namespace(use_vllm_generation=True, gpu_indices="0,1", gpu_index=1, vllm_gpu_indices="0")

    try:
        _validate_vllm_gpu_placement(args)
    except SystemExit as exc:
        assert "vLLM GPU overlap" in str(exc)
    else:
        raise AssertionError("expected GPU overlap validation to fail")


def test_validate_vllm_gpu_placement_allows_separate_gpus() -> None:
    args = Namespace(use_vllm_generation=True, gpu_indices="1", gpu_index=1, vllm_gpu_indices="0")

    _validate_vllm_gpu_placement(args)


class _TinyParamModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.lora_a = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=True)
        self.lora_b = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=True)


def test_collect_trainable_named_parameters_returns_only_trainable_cpu_tensors() -> None:
    model = _TinyParamModel()

    params = collect_trainable_named_parameters(model)

    assert sorted(params) == ["lora_a", "lora_b"]
    assert all(not tensor.requires_grad for tensor in params.values())
    assert all(tensor.device.type == "cpu" for tensor in params.values())
    assert params["lora_a"].item() == 2.0
    assert params["lora_b"].item() == 3.0


class _FakeTRLClient:
    def __init__(self) -> None:
        self.updated: list[tuple[str, torch.Tensor]] = []
        self.health_checked = False
        self.communicator_initialized = False

    def check_server(self) -> None:
        self.health_checked = True

    def init_communicator(self) -> None:
        self.communicator_initialized = True

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        assert self.communicator_initialized is True
        self.updated.append((name, weights))


def test_vllm_generation_client_syncs_trainable_parameters() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient()
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)
    model = _TinyParamModel()

    elapsed = client.sync_trainable_parameters(model)

    assert elapsed >= 0.0
    assert backend.communicator_initialized is True
    assert [name for name, _ in backend.updated] == ["lora_a", "lora_b"]
    assert all(tensor.device.type == "cpu" for _, tensor in backend.updated)


class _FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool):
        assert add_generation_prompt is True
        assert tokenize is True
        joined = "\n".join(item["content"] for item in messages)
        return [min(98, ord(char) % 100) for char in joined][-32:]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "decoded response"


class _LogprobModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input_ids, attention_mask=None, logits_to_keep=None):
        vocab_size = 128
        logits = self.weight * torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab_size, device=input_ids.device)
        return type("Output", (), {"logits": logits})


class _FakeVLLMClient:
    def __init__(self) -> None:
        self.prompts: list[list[int]] = []

    def generate(self, prompt_token_ids, *, max_tokens, temperature, top_p, top_k):
        assert isinstance(prompt_token_ids, str)
        self.prompts.append([ord(char) for char in prompt_token_ids[:4]])
        return [10, 11], "decoded response"


def test_vllm_shared_policy_generates_and_records_trace() -> None:
    from rl_training.policy import VLLMSharedPolicy

    client = _FakeVLLMClient()
    model = _LogprobModel()
    tokenizer = _FakeTokenizer()
    policy = VLLMSharedPolicy(
        model=model,
        tokenizer=tokenizer,
        vllm_client=client,
        system_prompt="system",
        max_prompt_length=32,
        max_completion_length=2,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
    )

    response = policy.generate(
        role=AgentRole.QUERY_RETRIEVER,
        question="Who?",
        state=RAGState(question="Who?"),
    )

    assert response == "decoded response"
    assert len(client.prompts) == 1
    assert len(policy.trace.actions) == 1
    action = policy.trace.actions[0]
    assert action.role == AgentRole.QUERY_RETRIEVER
    assert action.completion_ids == [10, 11]
    assert action.response == "decoded response"
    assert action.old_logprobs.shape == (2,)
    assert policy.timing["time_vllm_generate_seconds"] >= 0.0


def test_build_policy_uses_hf_policy_when_vllm_disabled() -> None:
    args = Namespace(
        use_vllm_generation=False,
        system_prompt="system",
        max_prompt_length=32,
        max_completion_length=2,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
    )

    policy = _build_policy(args, _LogprobModel(), _FakeTokenizer())

    assert isinstance(policy, HFSharedPolicy)


def test_train_on_rollouts_reports_optimizer_step_flag() -> None:
    model = _LogprobModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = type(
        "Args",
        (),
        {"gradient_accumulation_steps": 1, "clip_epsilon": 0.2, "kl_beta": 0.0},
    )()
    action = type(
        "Action",
        (),
        {
            "prompt_ids": [1, 2],
            "completion_ids": [3],
            "old_logprobs": torch.zeros(1),
        },
    )()
    rollouts = [{"advantage": 1.0, "actions": [action]}]

    metrics = _train_on_rollouts(
        rollouts=rollouts,
        train_model=model,
        raw_policy_model=model,
        ref_model=model,
        optimizer=optimizer,
        args=args,
        torch=torch,
        device=torch.device("cpu"),
        should_step=True,
    )

    assert metrics["did_optimizer_step"] is True
    assert "time_optimizer_step_seconds" in metrics


def test_load_rl_samples_reads_existing_extracted_files(tmp_path: Path) -> None:
    data_root = tmp_path / "rl"
    _write_jsonl(
        data_root / "hotpotqa_rl.jsonl",
        [
            {
                "qid": "q1",
                "dataset": "hotpotqa",
                "question": "Who directed The Tripper?",
                "answer": "David Arquette",
                "answer_aliases": ["Arquette"],
                "supporting_facts": [
                    {
                        "doc_id": "d1",
                        "title": "The Tripper",
                        "text": "The Tripper was directed by David Arquette.",
                    }
                ],
            },
            {
                "qid": "q2",
                "dataset": "hotpotqa",
                "question": "Bad row has no answer",
                "supporting_facts": [],
            },
        ],
    )

    samples, summary = load_rl_samples(data_root=data_root, data_files=[], max_samples=1)

    assert len(samples) == 1
    assert samples[0].qid == "q1"
    assert samples[0].answer == "David Arquette"
    assert samples[0].answer_aliases == ["Arquette"]
    assert samples[0].supporting_facts[0]["title"] == "The Tripper"
    assert summary["loaded_samples"] == 1
    assert summary["skipped_samples"] == 1
    assert summary["counts_by_dataset"] == {"hotpotqa": 1}


def test_load_rl_samples_default_files_ignore_corpus_jsonl(tmp_path: Path) -> None:
    data_root = tmp_path / "rl"
    _write_jsonl(
        data_root / "hotpotqa" / "hotpotqa_train.jsonl",
        [
            {
                "qid": "q1",
                "dataset": "hotpotqa",
                "question": "Who directed The Tripper?",
                "answer": "David Arquette",
                "supporting_facts": [{"title": "The Tripper", "text": "Directed by David Arquette."}],
            }
        ],
    )
    _write_jsonl(
        data_root / "hotpotqa" / "corpus.jsonl",
        [{"doc_id": "d1", "title": "The Tripper", "text": "Corpus rows are not RL samples."}],
    )

    samples, summary = load_rl_samples(data_root=data_root, data_files=[])

    assert len(samples) == 1
    assert summary["skipped_samples"] == 0
    assert summary["source_files"] == [str(data_root / "hotpotqa" / "hotpotqa_train.jsonl")]


def test_run_train_grpo_script_derives_gpu_visibility_from_yaml() -> None:
    script = Path("scripts/run_train_grpo.sh").read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES:-0,1" not in script
    assert "CONFIG_PATH=" in script
    assert "yaml.safe_load" in script
    assert "NPROC_PER_NODE" in script
    assert 'export CUDA_VISIBLE_DEVICES="${YAML_GPU_INDICES}"' in script
    assert 'export MACORAG_SILENT_RETRIEVAL="${MACORAG_SILENT_RETRIEVAL:-1}"' in script
    assert "--nproc_per_node=${NPROC_PER_NODE}" in script


def test_run_grpo_vllm_server_script_uses_vllm_gpu_and_trl_server() -> None:
    script = Path("scripts/run_grpo_vllm_server.sh").read_text(encoding="utf-8")

    assert "CONFIG_PATH=" in script
    assert "vllm_gpu_indices" in script
    assert 'export CUDA_VISIBLE_DEVICES="${YAML_VLLM_GPU_INDICES}"' in script
    assert "trl vllm-serve" in script
    assert "--model" in script
    assert "--host" in script
    assert "--port" in script
    assert "--tensor-parallel-size" in script
    assert "--gpu-memory-utilization" in script


def test_write_train_event_records_sample_progress(tmp_path: Path) -> None:
    event_path = tmp_path / "train_events.jsonl"

    _write_train_event(
        event_path,
        event="sample_start",
        epoch=1,
        sample_index=4,
        sample_total=12,
        sample_qid="q5",
        sample_dataset="2wiki",
        step=3,
        group_index=0,
    )

    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "event": "sample_start",
            "epoch": 1,
            "sample": 5,
            "sample_total": 12,
            "qid": "q5",
            "dataset": "2wiki",
            "step": 3,
            "group_index": 0,
        }
    ]


def test_compute_answer_f1_uses_normalized_token_overlap_and_aliases() -> None:
    assert compute_answer_f1("the david  arquette!", "David Arquette", []) == 1.0
    assert compute_answer_f1("Arquette", "David Arquette", ["Arquette"]) == 1.0
    assert compute_answer_f1("David", "David Arquette", []) == 2 / 3
    assert compute_answer_f1("", "David Arquette", []) == 0.0


def test_compute_rl_rewards_scores_query_evidence_and_final_answer() -> None:
    rollout = {
        "trajectory": [
            {
                "query_retriever": {
                    "sub_goal": "find director",
                    "query": "The Tripper director",
                },
                "observation": {
                    "passages": [
                        {
                            "passage_id": 0,
                            "doc_id": "d1",
                            "title": "The Tripper",
                            "text": "The Tripper was directed by David Arquette.",
                        },
                        {"passage_id": 1, "doc_id": "d2", "title": "Noise", "text": "Irrelevant."},
                    ]
                },
                "update_evidence": {"selected_passage_ids": [0]},
                "answer": {"can_answer": True, "answer": "David Arquette"},
            }
        ],
        "parse_errors": [],
        "final_answer": "David Arquette",
    }
    sample = {
        "answer": "David Arquette",
        "answer_aliases": [],
        "supporting_facts": [
            {
                "doc_id": "d1",
                "title": "The Tripper",
                "text": "The Tripper was directed by David Arquette.",
            }
        ],
    }

    rewards = compute_rl_rewards(rollout=rollout, sample=sample)

    assert rewards["query_reward"] > 0.0
    assert rewards["evidence_reward"] > 0.0
    assert rewards["answer_f1"] == 1.0
    assert rewards["answer_reward"] == 1.0
    assert rewards["total"] > 2.0


def test_compute_rl_rewards_penalizes_wrong_premature_multihop_answer() -> None:
    rollout = {
        "trajectory": [
            {
                "query_retriever": {"sub_goal": "find director", "query": "Bullitt film director"},
                "observation": {
                    "passages": [
                        {
                            "passage_id": 0,
                            "doc_id": "d1",
                            "title": "Bullitt",
                            "text": "Bullitt was directed by Peter Yates.",
                        }
                    ]
                },
                "update_evidence": {"selected_passage_ids": [0]},
                "answer": {"can_answer": True, "answer": "London"},
            }
        ],
        "parse_errors": [],
        "final_answer": "London",
    }
    sample = {
        "answer": "Aldershot",
        "answer_aliases": [],
        "supporting_facts": [
            {"doc_id": "d1", "title": "Bullitt", "text": "Bullitt was directed by Peter Yates."},
            {
                "doc_id": "d2",
                "title": "Peter Yates",
                "text": "Peter Yates was born in Aldershot, Hampshire.",
            },
        ],
    }

    rewards = compute_rl_rewards(rollout=rollout, sample=sample)

    assert rewards["answer_f1"] == 0.0
    assert rewards["support_facts_required"] == 2.0
    assert rewards["support_facts_covered"] == 1.0
    assert rewards["premature_answer_penalty"] == -1.0


def test_compute_rl_rewards_does_not_penalize_correct_or_sufficient_multihop_answer() -> None:
    sample = {
        "answer": "Aldershot",
        "answer_aliases": [],
        "supporting_facts": [
            {"doc_id": "d1", "title": "Bullitt", "text": "Bullitt was directed by Peter Yates."},
            {
                "doc_id": "d2",
                "title": "Peter Yates",
                "text": "Peter Yates was born in Aldershot, Hampshire.",
            },
        ],
    }
    rollout = {
        "trajectory": [
            {
                "query_retriever": {"sub_goal": "find director", "query": "Bullitt film director"},
                "observation": {
                    "passages": [
                        {
                            "passage_id": 0,
                            "doc_id": "d1",
                            "title": "Bullitt",
                            "text": "Bullitt was directed by Peter Yates.",
                        },
                        {
                            "passage_id": 1,
                            "doc_id": "d2",
                            "title": "Peter Yates",
                            "text": "Peter Yates was born in Aldershot, Hampshire.",
                        },
                    ]
                },
                "update_evidence": {"selected_passage_ids": [0, 1]},
                "answer": {"can_answer": True, "answer": "Aldershot"},
            }
        ],
        "parse_errors": [],
        "final_answer": "Aldershot",
    }

    rewards = compute_rl_rewards(rollout=rollout, sample=sample)

    assert rewards["support_facts_required"] == 2.0
    assert rewards["support_facts_covered"] == 2.0
    assert rewards["premature_answer_penalty"] == 0.0


def test_compute_grpo_loss_uses_advantages_clipping_and_kl() -> None:
    current = torch.log(torch.tensor([[0.6, 0.4], [0.2, 0.8]], dtype=torch.float32))
    old = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32))
    reference = torch.log(torch.tensor([[0.55, 0.45], [0.4, 0.6]], dtype=torch.float32))
    mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)

    loss, metrics = compute_grpo_loss(
        current_logprobs=current,
        old_logprobs=old,
        ref_logprobs=reference,
        action_mask=mask,
        advantages=advantages,
        clip_epsilon=0.2,
        kl_beta=0.1,
    )

    assert loss.requires_grad is False
    assert torch.isfinite(loss)
    assert metrics["policy_loss"] != 0.0
    assert metrics["kl"] >= 0.0
    assert metrics["loss"] == float(loss.item())


def test_policy_generate_disables_cache_for_gradient_checkpointing(monkeypatch) -> None:
    class DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert add_generation_prompt is True
            assert tokenize is True
            return [1, 2, 3]

        def decode(self, token_ids, skip_special_tokens=True):
            return "<answer>{\"can_answer\":true,\"answer\":\"Ada\"}</answer>"

    class DummyModel:
        def __init__(self) -> None:
            self.generate_kwargs = None
            self.parameter = torch.nn.Parameter(torch.tensor(1.0))

        def parameters(self):
            return iter([self.parameter])

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return torch.tensor([[1, 2, 3, 4, 9]])

    def fake_logprobs(**kwargs):
        return torch.zeros(len(kwargs["completion_ids"]))

    model = DummyModel()
    monkeypatch.setattr("rl_training.policy.sequence_logprobs", fake_logprobs)
    policy = HFSharedPolicy(
        model=model,
        tokenizer=DummyTokenizer(),
        system_prompt="system",
        max_prompt_length=16,
        max_completion_length=8,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )

    policy.generate(
        role=AgentRole.ANSWER_GENERATOR,
        question="Who?",
        state=RAGState(question="Who?"),
    )

    assert model.generate_kwargs["use_cache"] is False


def test_sequence_logprobs_keeps_only_completion_logits() -> None:
    class DummyOutput:
        def __init__(self, logits):
            self.logits = logits

    class DummyModel:
        def __init__(self) -> None:
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            input_ids = kwargs["input_ids"]
            logits_to_keep = kwargs["logits_to_keep"]
            vocab_size = 16
            logits = torch.full((1, logits_to_keep, vocab_size), -20.0)
            labels = torch.nn.functional.pad(input_ids[:, 1:], (0, 1), value=-100)
            kept_labels = labels[:, -logits_to_keep:]
            for index, token_id in enumerate(kept_labels[0].tolist()):
                if token_id >= 0:
                    logits[0, index, token_id] = 20.0
            return DummyOutput(logits)

    model = DummyModel()
    prompt_ids = [1, 2, 3, 4, 5]
    completion_ids = [6, 7, 8]

    logprobs = sequence_logprobs(
        model=model,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        device=torch.device("cpu"),
    )

    assert model.kwargs["logits_to_keep"] == len(completion_ids) + 1
    assert logprobs.shape == (len(completion_ids),)
    assert torch.all(logprobs > -1e-4)


def test_train_on_rollouts_backprops_each_action_to_release_graphs(monkeypatch) -> None:
    class DummyAction:
        def __init__(self, value: float) -> None:
            self.prompt_ids = [1, 2]
            self.completion_ids = [3]
            self.old_logprobs = torch.tensor([value], dtype=torch.float32)

    class DummyOptimizer:
        def __init__(self) -> None:
            self.steps = 0
            self.zero_grad_calls = 0

        def step(self) -> None:
            self.steps += 1

        def zero_grad(self, set_to_none: bool = False) -> None:
            assert set_to_none is True
            self.zero_grad_calls += 1

    class Args:
        gradient_accumulation_steps = 1
        clip_epsilon = 0.2
        kl_beta = 0.02

    backward_calls = []

    def fake_sequence_logprobs(**kwargs):
        value = float(kwargs["completion_ids"][0])
        return torch.tensor([value], dtype=torch.float32, requires_grad=True)

    original_backward = torch.Tensor.backward

    def counting_backward(self, *args, **kwargs):
        backward_calls.append(float(self.detach().item()))
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr("rl_training.train_grpo_macorag.sequence_logprobs", fake_sequence_logprobs)
    monkeypatch.setattr(torch.Tensor, "backward", counting_backward)

    metrics = _train_on_rollouts(
        rollouts=[
            {
                "advantage": 1.0,
                "actions": [DummyAction(3.0), DummyAction(4.0)],
            }
        ],
        train_model=object(),
        raw_policy_model=object(),
        ref_model=object(),
        optimizer=DummyOptimizer(),
        args=Args(),
        torch=torch,
        device=torch.device("cpu"),
        should_step=True,
    )

    assert len(backward_calls) == 2
    assert metrics["loss"] != 0.0
    assert "time_backward_seconds" in metrics
    assert "time_optimizer_step_seconds" in metrics
    assert "time_train_seconds" not in metrics
