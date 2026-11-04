# SPDX-FileCopyrightText: 2026 Jonas Kunze <jonas.kunze@htwk-leipzig.de>
# SPDX-FileCopyrightText: 2026 Jonas Wagner <jonas.wagner@htwk-leipzig.de>
#
# SPDX-License-Identifier: MIT

"""WSE Clarifier.

Always asks exactly one clarifying question, then selects among several final
implementations by execution. Before asking, the LLM names the main defect of the
task description, and that note guides the question. After the answer, three
independent implementations are generated. Asserts from the task's doctest examples
and from the user's answer are run against them, and the implementation that passes
the most asserts is returned.

Team: WiSE
Team Members: Jonas Kunze, Jonas Wagner, Jonas Gwozdz, Parnian Hajian, Dennis Schiese, Andreas Both
Main Contact: jonas.kunze@htwk-leipzig.de
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)

N_FINAL = 3  # independent final implementations to select from
MAX_ANSWER_ASSERTS = 4
REPAIR_ATTEMPTS = 2  # retries when a response has no usable fenced solution

# Prompts ------------------------------------------------------------------------------

DEFECT = """
Classify the main defect of this coding task description.

Task (function to implement: `{entry_point}`):
{prompt}

Options: clear, ambiguous, incomplete, contradictory, vague_wording, underspecified, formatting.
Reply with exactly one line:
DEFECT: <option> | <one short sentence naming the problematic part>
""".strip()

DEFECT_HINT = "\nA reviewer noted this about the task: {note}\n"

RULES = (
    "The question must target exactly one fact, admit one unambiguous answer, not be answerable "
    "from the task text, and must not ask for the algorithm, the implementation or the test cases."
)

QUESTION = """
You are reviewing a coding task before implementing it. You may ask the user exactly one
clarifying question. The user knows the intended behaviour that the hidden tests check.

Task (function to implement: `{entry_point}`):
{prompt}
{hint}
Find the single fact whose wrong guess would most likely make a correct-looking
implementation fail the user's tests: a missing or contradictory condition, the expected
return type or format, behaviour on an edge case, the meaning of a vague term, the order
or meaning of the arguments. Prefer the fact that changes the most outputs.

Reply with exactly one line:
QUESTION: <one specific question about exactly one fact>

{rules}
""".strip()

CLARIFICATION = """
The user was asked a clarifying question and answered:
Q: {question}
A: {answer}

Follow that answer exactly, even where it contradicts the task text.
"""

SOLVE = """
Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}
{clarification}
Enclose your solution in ```python and ```.
""".strip()

REPAIR = """
Your response did not contain a valid fenced Python solution.
Return only the complete implementation of `{entry_point}`, enclosed in ```python and ```.
""".strip()

ANSWER_ASSERTS = """
Task (function to implement: `{entry_point}`):
{prompt}
{clarification}
Write up to 4 assert statements that check the behaviour the user's answer fixes, for
example `assert {entry_point}(...) == ...`. Only include cases you are certain about.
Return only the asserts, enclosed in ```python and ```.
""".strip()

EMPTY = "def {entry_point}(*args, **kwargs):\n    raise NotImplementedError\n"

# Runs every candidate against every assert in one sandbox call and prints the pass counts.
ASSERT_RUNNER = r"""
import json, signal
CODES = json.loads({codes!r})
ASSERTS = json.loads({asserts!r})
def handler(signum, frame):
    raise TimeoutError()
signal.signal(signal.SIGALRM, handler)
scores = []
for code in CODES:
    ns = {{}}
    try:
        exec(code, ns)
    except Exception:
        scores.append(-1)
        continue
    ok = 0
    for a in ASSERTS:
        try:
            signal.alarm(2)
            exec(a, ns)
            ok += 1
        except BaseException:
            pass
        finally:
            signal.alarm(0)
    scores.append(ok)
print("@@SCORES@@" + json.dumps(scores))
"""

# Helpers ------------------------------------------------------------------------------


def fenced(response: str) -> str | None:
    """The first ```python block, or the first ``` block, or None."""
    m = re.search(r"```python[^\n]*\n(.*?)```", response, flags=re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```[^\n]*\n(.*?)```", response, flags=re.DOTALL)
    return m.group(1) if m else None


def defines(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry_point
        for n in ast.walk(tree)
    )


def is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    t = node.test
    return (
        isinstance(t, ast.Compare)
        and isinstance(t.left, ast.Name)
        and t.left.id == "__name__"
        and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in t.comparators)
    )


def clean(code: str, entry_point: str) -> str:
    """Drop demo code (main guard, top-level calls, asserts, loops); keep definitions."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    keep = []
    for node in tree.body:
        if is_main_guard(node):
            continue
        if isinstance(node, (ast.Assert, ast.For, ast.While, ast.AsyncFor)):
            continue
        if isinstance(node, ast.Expr) and not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            continue
        keep.append(node)
    tree.body = keep
    out = ast.unparse(tree)
    if not defines(out, entry_point):
        funcs = [n.name for n in keep if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(funcs) == 1:
            out += f"\n\n{entry_point} = {funcs[0]}\n"
    return out


def parse_question(text: str) -> str | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if any(ln.upper().strip("`*. ") == "NO_QUESTION" for ln in lines):
        return None
    for ln in lines:
        stripped = ln.lstrip("`*-# ")
        if stripped.upper().startswith("QUESTION:"):
            q = stripped.split(":", 1)[1].strip().strip("`*")
            if q:
                return q
    marked = [ln for ln in lines if ln.endswith("?")]
    return marked[0] if len(marked) == 1 else None


def doctest_asserts(prompt: str, entry_point: str) -> list[str]:
    """`>>> f(x)` followed by an expected line, turned into asserts."""
    lines = prompt.splitlines()
    out = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith(">>>") and f"{entry_point}(" in s and i + 1 < len(lines):
            call = s[3:].strip()
            expected = lines[i + 1].strip()
            if expected and not expected.startswith(">>>"):
                out.append(f"assert {call} == {expected}")
    return out


def marker_json(output: str, marker: str) -> Any:
    for ln in output.splitlines():
        if ln.startswith(marker):
            try:
                return json.loads(ln[len(marker) :])
            except json.JSONDecodeError:
                return None
    return None


# Algorithm ----------------------------------------------------------------------------


class WSEClarifier(ClarificationAlgorithmBase):
    def ask_llm(self, env: ClarificationEnvironment, content: str | list) -> str:
        try:
            return env.llm(content)
        except LimitsExceededException:
            return ""

    def code_from(
        self, env: ClarificationEnvironment, prompt_text: str, entry_point: str
    ) -> str | None:
        messages = [{"role": "user", "content": prompt_text}]
        for _ in range(REPAIR_ATTEMPTS + 1):
            response = self.ask_llm(env, messages)
            if not response:
                return None
            code = fenced(response)
            if code and defines(code, entry_point):
                return code
            if code:
                fixed = clean(code, entry_point)
                if defines(fixed, entry_point) or f"{entry_point} =" in fixed:
                    return fixed
            messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content": REPAIR.format(entry_point=entry_point)},
            ]
        return code if code else None

    def defect_hint(self, env: ClarificationEnvironment, prompt: str, entry_point: str) -> str:
        text = self.ask_llm(env, DEFECT.format(prompt=prompt, entry_point=entry_point))
        line = next(
            (ln.strip() for ln in text.splitlines() if ln.strip().upper().startswith("DEFECT:")), ""
        )
        note = line[7:].strip()
        return DEFECT_HINT.format(note=note) if note else ""

    def question(self, env: ClarificationEnvironment, prompt: str, entry_point: str) -> str | None:
        hint = self.defect_hint(env, prompt, entry_point)
        text = self.ask_llm(
            env, QUESTION.format(prompt=prompt, entry_point=entry_point, hint=hint, rules=RULES)
        )
        return parse_question(text)

    def checks(
        self, env: ClarificationEnvironment, prompt: str, entry_point: str, clarification: str
    ) -> list[str]:
        asserts = doctest_asserts(prompt, entry_point)
        if clarification:
            text = self.ask_llm(
                env,
                ANSWER_ASSERTS.format(
                    prompt=prompt, entry_point=entry_point, clarification=clarification
                ),
            )
            block = fenced(text) or ""
            asserts += [
                ln.strip() for ln in block.splitlines() if ln.strip().startswith("assert ")
            ][:MAX_ANSWER_ASSERTS]
        return asserts

    def select(self, env: ClarificationEnvironment, finals: list[str], asserts: list[str]) -> str:
        """The first implementation among those that pass the most asserts."""
        scores = None
        if asserts:
            script = ASSERT_RUNNER.format(codes=json.dumps(finals), asserts=json.dumps(asserts))
            try:
                scores = marker_json(env.exec_code(script), "@@SCORES@@")
            except Exception:
                scores = None
            if not scores or len(scores) != len(finals):
                scores = None
        scores = scores or [0] * len(finals)
        return finals[scores.index(max(scores))]

    def final_code(
        self, env: ClarificationEnvironment, prompt: str, entry_point: str, clarification: str
    ) -> str:
        text = SOLVE.format(prompt=prompt, entry_point=entry_point, clarification=clarification)
        finals = []
        for _ in range(N_FINAL):
            code = self.code_from(env, text, entry_point)
            if code:
                finals.append(clean(code, entry_point))
        if not finals:
            return EMPTY.format(entry_point=entry_point)
        if len(finals) == 1:
            return finals[0]
        asserts = self.checks(env, prompt, entry_point, clarification)
        return self.select(env, finals, asserts)

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        prompt, entry_point = problem["prompt"], problem["entry_point"]
        clarification = ""
        try:
            if env.can_ask():
                question = self.question(env, prompt, entry_point)
                if question:
                    try:
                        answer = env.ask_human(question)
                        clarification = CLARIFICATION.format(question=question, answer=answer)
                    except TooManyQuestionException:
                        pass
            code = self.final_code(env, prompt, entry_point, clarification)
        except LimitsExceededException:
            code = EMPTY.format(entry_point=entry_point)
        return f"```python\n{code}\n```"
