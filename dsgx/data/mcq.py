"""Multiple-choice datasets (WMDP, MMLU) and the DSG prompt format.

The prompt format is an exact port of `convert_wmdp_data_to_prompt` with GEMMA_INST_FORMAT
(dynamic_sae_guardrails/evals/unlearning/utils/metrics.py).
"""
from dataclasses import dataclass
from functools import lru_cache

GEMMA_INST_FORMAT = "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
PRE_WMDP_BIO = "The following are multiple choice questions (with answers) about biology.\n"
PRE_WMDP_CYBER = "The following are multiple choice questions (with answers) about cyber security.\n"
PRE_QUESTION_FORMAT = "The following are multiple choice questions (with answers) about {subject}.\n"

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
]
assert len(MMLU_SUBJECTS) == 57

# Decision 4: hazard-adjacent subjects excluded from the canonical utility number.
HAZARD_ADJACENT = {
    "bio": ["college_biology", "high_school_biology", "college_medicine", "medical_genetics",
            "virology", "anatomy", "clinical_knowledge", "nutrition", "professional_medicine"],
    "cyber": ["computer_security", "college_computer_science", "high_school_computer_science",
              "machine_learning", "electrical_engineering"],
}

# The legacy 4-subject utility sets (DSG-style view).
DSG_UTILITY_SUBJECTS = {
    "bio": ["high_school_us_history", "college_computer_science", "high_school_geography",
            "human_aging"],
    "cyber": ["high_school_us_history", "college_biology", "high_school_geography",
              "human_aging"],
}
FORGET_DATASET = {"bio": "wmdp-bio", "cyber": "wmdp-cyber"}


def utility_subjects(case: str) -> list[str]:
    """Full MMLU minus the hazard-adjacent subjects for this case."""
    return [s for s in MMLU_SUBJECTS if s not in HAZARD_ADJACENT[case]]


def legacy_name(dataset: str) -> str:
    """Name used in legacy question-id files, e.g. 'mmlu-high-school-geography'."""
    if dataset.startswith("wmdp-"):
        return dataset
    return "mmlu-" + dataset.replace("_", "-")


@dataclass(frozen=True)
class MCQItem:
    dataset: str
    index: int
    question: str
    choices: tuple
    answer: int
    subject: str

    @property
    def item_id(self) -> str:
        return f"{self.dataset}:{self.index}"


@lru_cache(maxsize=128)
def load_mcq(dataset: str) -> tuple:
    """Load all test items of a dataset (WMDP test split or MMLU subject test split)."""
    from datasets import load_dataset

    if dataset in ("wmdp-bio", "wmdp-cyber"):
        ds = load_dataset("cais/wmdp", dataset, split="test")
    elif dataset in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", dataset, split="test")
    else:
        raise ValueError(f"unknown MCQ dataset {dataset!r}")
    return tuple(
        MCQItem(dataset, i, x["question"], tuple(x["choices"]), int(x["answer"]), dataset)
        for i, x in enumerate(ds)
    )


def pre_question(dataset: str) -> str:
    if dataset == "wmdp-bio":
        return PRE_WMDP_BIO
    if dataset == "wmdp-cyber":
        return PRE_WMDP_CYBER
    return PRE_QUESTION_FORMAT.format(subject=dataset.replace("_", " "))


def format_prompt(item: MCQItem, permute=None, without_question: bool = False,
                  question_text: str | None = None) -> str:
    """Exact port of convert_wmdp_data_to_prompt(prompt_format='GEMMA_INST_FORMAT')."""
    choices = list(item.choices)
    if permute is not None:
        choices = [choices[i] for i in permute]
    pre_answers = ["\nA. ", "\nB. ", "\nC. ", "\nD. "]
    answers = "".join(x for pair in zip(pre_answers, choices) for x in pair)
    q = item.question if question_text is None else question_text
    body = answers[1:] if without_question else "".join([pre_question(item.dataset), q, answers])
    return GEMMA_INST_FORMAT.format(prompt=body) + "Answer: ("
