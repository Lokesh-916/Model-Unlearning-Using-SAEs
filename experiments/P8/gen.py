"""P8 generation task (GPU): experiments.E5.gen.task UNCHANGED (frozen v2 gate on prompt + every token, arms A / B,
frozen-setting guard), with only its prompt builder swapped for experiments.P8.common.wrapped_prompt. The condition
travels as E5's `attack` argument ('none' or '<wrapper>|<none|pad400|split3>'), so it is part of the item-checkpoint
key and of the run config. Generations stay under $DSG_PRIVATE; items.parquet has ids, hashes and scores only."""
import experiments.E5.gen as e5gen

from experiments.P8.common import wrapped_prompt


def _builder(question, attack, item_id):
    return wrapped_prompt(question, attack, item_id)


def task(ctx):
    e5gen.open_attack_prompt = _builder
    return e5gen.task(ctx)
