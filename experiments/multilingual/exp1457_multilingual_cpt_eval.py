# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluate the multilingual CPT continuation (exp1457) against multilingual LM Eval Harness tasks.
"""

from experiments.evals.engine_configs import DEFAULT_LM_EVAL_MODEL_KWARGS
from experiments.evals.evals import default_eval, evaluate_lm_evaluation_harness, extract_model_name_and_path
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8
from experiments.evals.task_configs import (
    MULTILINGUAL_LM_EVAL_GENERATIVE_TASKS,
    MULTILINGUAL_LM_EVAL_LOGPROB_TASKS,
)
from experiments.multilingual.exp1457_multilingual_cpt import multilingual_cpt_8b_fineweb2_hq
from marin.execution.executor import executor_main

multilingual_lm_eval_logprob = default_eval(
    step=multilingual_cpt_8b_fineweb2_hq,
    resource_config=SINGLE_TPU_V5p_8,
    evals=MULTILINGUAL_LM_EVAL_LOGPROB_TASKS,
)

multilingual_eval_steps = [multilingual_lm_eval_logprob]

# if MULTILINGUAL_LM_EVAL_GENERATIVE_TASKS:
#     model_name, model_path = extract_model_name_and_path(multilingual_cpt_8b_fineweb2_hq)
#     multilingual_lm_eval_generative = evaluate_lm_evaluation_harness(
#         model_name,
#         model_path,
#         list(MULTILINGUAL_LM_EVAL_GENERATIVE_TASKS),
#         engine_kwargs=DEFAULT_LM_EVAL_MODEL_KWARGS,
#         resource_config=SINGLE_TPU_V5p_8,
#     )
#     multilingual_eval_steps.append(multilingual_lm_eval_generative)

if __name__ == "__main__":
    executor_main(steps=multilingual_eval_steps)
