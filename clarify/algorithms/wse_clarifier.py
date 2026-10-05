# SPDX-FileCopyrightText: 2026 Jonas Kunze <jonas.kunze@htwk-leipzig.de>
#
# SPDX-License-Identifier: MIT

"""WSE Clarifier.

Registration placeholder. For now the LLM is asked for a solution directly;
if it abstains on its own, its response is sent to the user as a clarifying
question. The final algorithm replaces this before the submission deadline.

Team: WiSE
Team Members: Jonas Kunze, Jonas Wagner, Parnian Hajian, Dennis Schiese
Main Contact: jonas.kunze@htwk-leipzig.de
"""

from __future__ import annotations

from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

PROMPT_TEMPLATE = """
Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}

Enclose your solution in ```python and ```.
""".strip()


class WSEClarifier(ClarificationAlgorithmBase):
    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        messages = [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"]).replace(
                    "{entry_point}", problem["entry_point"]
                ),
            }
        ]
        response = env.llm(messages)
        messages += [{"role": "assistant", "content": response}]

        while True:
            try:
                return _validate_and_parse_evalplus_result(response)
            except ValueError:
                # No implementation in the response, so treat it as a clarifying question.
                try:
                    answer = env.ask_human(response)
                    messages += [{"role": "user", "content": answer}]
                    response = env.llm(messages)
                    messages += [{"role": "assistant", "content": response}]
                except TooManyQuestionException:
                    return response
