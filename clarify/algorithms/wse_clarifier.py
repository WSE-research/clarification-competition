# SPDX-FileCopyrightText: 2026 Andreas Both <andreas.both@htwk-leipzig.de>
# SPDX-FileCopyrightText: 2026 Jonas Kunze <jonas.kunze@htwk-leipzig.de>
#
# SPDX-License-Identifier: MIT

"""WSE Clarifier.

Asks a clarifying question and checks everything else by running code.

1. Five drafts are written, each under a different instruction, and run on the task's examples
   and on calls that the model writes.
2. If even the best draft fails most examples, text and examples disagree, and the question is
   about that. Otherwise it describes the best draft with two calls it really made and asks
   whether that is the intended behaviour.
3. The answer is turned into checks: the calls whose results it settles.
4. Six candidates are written and ranked by those checks, then by the task's examples, then by
   the fewest errors, then by agreement with the other candidates.
5. If a further question is allowed and the candidates still differ, the next question describes
   the chosen candidate with calls on which they differ, and steps 3 and 4 follow its answer.

Team: WiSE
Team Members: Jonas Kunze, Jonas Wagner, Jonas Gwozdz, Parnian Hajian, Dennis Schiese, Andreas Both
Main Contact: jonas.kunze@htwk-leipzig.de
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)

WRITE_ATTEMPTS = 3  # tries when a response contains no code block
SELECTION_CANDIDATES = 6  # implementations written after an answer
SELECTION_CALLS = 12  # calls written by the model to compare implementations
CONTRACT = ("args", "returns", "rule")
# How much a call counts: stated in an answer, in the task, neither.
ANSWER, EXAMPLE, PROBE = 2, 1, 0
Outputs = list[str | None] | None
Key = tuple[int, int, int, int] | None

SYSTEM_PROMPT = (
    "You are a careful senior Python engineer. You write correct, self-contained "
    "standard-library Python and you follow the user's stated requirements exactly."
)

GENERATE_TEMPLATE = """
Write a self-contained Python implementation of the function `{entry_point}` for the task below.

## Task
{prompt}
{clarifications}
## Rules
- Define a top-level function named exactly `{entry_point}`. If the task does not spell out the parameters, choose the most natural signature for the description (the inputs in the order they are mentioned).{signature_rule}
- The user's clarifications are authoritative: follow them exactly, even where they contradict the task text or its examples.
- A requirement that contradicts the obvious purpose of `{entry_point}` (an impossible or irrelevant constraint, a widened input, an extra output, a changed return type) was probably added by mistake: leave it out unless a clarification confirms it.
- Use only the Python standard library. No input(), no printing, no example usage, no tests.

Return the complete implementation in a single ```python block.
""".strip()

CALLS_TEMPLATE = """
List {count} different calls of `{entry_point}` that the user's tests could make for the task below: valid inputs of the intended kind, from typical cases to edge cases (empty or minimal inputs, ties, boundaries).

## Task
{prompt}
{clarifications}
Write each call as one Python expression on its own line, like `{entry_point}(...)`, with literal arguments only. No expected results, no comments, no numbering, no other text.
""".strip()

# Runs every candidate on the same calls in ONE script (the official exec_code starts a
# container per call and allows 30 s): 2 s to load a candidate, 1 s per call, and calls
# after 20 s are skipped. Outputs are "=<repr>" for values and "!<error>" for errors.
HARNESS_MARK = "WSE_RESULT:"
HARNESS = """
import json
import signal
import time


def on_alarm(signum, frame):
    raise TimeoutError


signal.signal(signal.SIGALRM, on_alarm)


def normalized(value):
    return "=" + (repr(round(value, 6)) if isinstance(value, float) else repr(value)[:300])


def limited(seconds, function, *args):
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return function(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


sources = {sources}
calls = {calls}
expected = {expected}
entry = {entry_point!r}
aliases = {aliases}
started = time.time()
expectations = []
for text in expected:
    try:
        expectations.append(None if text is None else normalized(limited(1.0, eval, text, {{}})))
    except BaseException:
        expectations.append(None)
report = []
for source in sources:
    space = {{}}
    try:
        limited(2.0, exec, source, space)
        for alias in aliases:
            space[alias] = space[entry]
    except BaseException as error:
        report.append({{"error": type(error).__name__}})
        continue
    outputs = []
    for call in calls:
        if time.time() - started > 20:
            outputs.append(None)
            continue
        try:
            outputs.append(normalized(limited(1.0, eval, call, space)))
        except BaseException as error:
            outputs.append("!" + type(error).__name__)
    report.append({{"outputs": outputs}})
print("WSE_RESULT:" + json.dumps({{"expected": expectations, "candidates": report}}))
""".strip()

GAP_TEMPLATE = """
You may ask the user who wrote the task below exactly one clarifying question before delivering the implementation of `{entry_point}`.

## Task
{prompt}

## Situation
{situation}

## Requirements for the question
- If the description offers alternatives ("A or B") or a vague condition, ask which option is meant, naming the options as a choice, for example "Should `{entry_point}` return the sum or the product of the odd digits?".
- Otherwise ask about the one fact that most likely changes the expected outputs of the user's tests, offering concrete alternatives where possible.
- Ask about exactly one fact. Never ask about invalid inputs or error handling, and never for code, examples or test cases. Keep it under 40 words.

Return only the question, nothing else.
""".strip()

EXAMPLES_CONFLICT = "An implementation that follows the text does not reproduce the examples in the task. Either the examples are wrong, or the intended behaviour is a different one."
TEXT_GAP = "Information may be missing from this description: a whole sentence (a special return value, a tie rule, an ordering rule, a bound), a qualifier, the examples, or the core rule itself."

CONTRACT_TEMPLATE = """
Describe what the implementation of `{entry_point}` below does, for the user who wrote the task. Describe this implementation, not the task.

## Task
{prompt}

## Implementation
```python
{code}
```

Answer with one JSON object and no other text:
{{"args": "<the arguments it takes, in order, with their kind, for example 'a list of integers and an integer k'>", "returns": "<what it returns, with its type and format, for example 'the number of pairs as an int'>", "rule": "<the rule it computes, in one short plain sentence>"}}
""".strip()

CONTRACT_QUESTION = "About {entry_point}: my current version takes {args} and returns {returns}; {rule}{example} Is that what you intend, or should it take different inputs, behave differently or return a different format?"

TRIVIAL_VALUES = {"[]", "()", "''", '""', "{}", "set()", "None", "0", "0.0"}

# The environment's fixed replies when it cannot or will not answer.
NON_ANSWER_RE = re.compile(r"^\W*(cannot answer the given question|irrelevant question)\W*$", re.I)
MARKUP_HINT = "\n\n(Do not write an opening square bracket directly followed by a slash; write `[\\/` in regular expressions instead.)"

ANSWER_CHECKS_TEMPLATE = """
The user who wrote the task below answered a clarifying question about `{entry_point}`. Write down the calls of `{entry_point}` whose results the answer settles.

## Task
{prompt}

## Question
{question}

## Answer
{answer}

## Rules
- Include a call only if the answer states its result, or if the question gives an example call with its result and the answer confirms it.
- Do not include results you infer from the task text alone, and nothing the answer leaves open or rejects.
- Write each call as one line `{entry_point}(<literal arguments>) == <literal result>`, at most 5 lines. If the answer settles no call, write NONE.

Return only these lines.
""".strip()


INPUT_RULE = "\n- Accept any input of the kind the description names: a list or a tuple where a sequence is meant, unsorted, empty or with duplicates. Do not validate inputs and do not raise errors for them. Where the description leaves the form of an input open (one number, or a sequence of numbers), take the form that the function's name suggests."

DRAFT_FRAMINGS = [
    "\n- Implement exactly what each sentence of the task says, read literally.",
    "\n- Before writing code, decide what the person who wrote this short exercise most likely means, using the function name and any examples.",
    "\n- The task describes a short, common programming exercise: implement the usual solution of that exercise.",
    "\n- Make sure the implementation reproduces every example in the task; where the text is vague, let the examples decide.",
]

GENERATE_LENGTH_RULE = "\n- The task shows no function header. Take the inputs the description names, in the order it mentions them. Do not add a parameter for the length of a sequence or for start and end indices: such arguments of the tests are handled automatically."

CLARIFICATIONS_HEADING = "\n## Clarifications from the user (authoritative; they override the task text where they conflict)"
DRAFT_NOTE_QUESTION = " (the assistant's description of its earlier draft, not a statement of the user; only what the answer confirms counts)"
DRAFT_NOTE_CORRECTION = "\n(This answer corrects the earlier draft described in Q{i}. Rebuild every part it touches from the task text and the answer, and keep nothing of the draft there.)"

# Every implementation runs with these names, because models often drop the imports.
PRELUDE = """from typing import *
import bisect, collections, copy, datetime, decimal, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string, sys

"""

# Appended to the returned code: a failing call is retried with the arguments in another form,
# because the tests of a task may pass them differently from the signature the model chose.
ADAPTER = """


def wse_variants(args):
    orders = list(itertools.permutations(args)) if len(args) <= 4 else [args]
    first = args[0] if args else None
    forms = [args[:i] for i in range(len(args) - 1, 0, -1)] + [(list(args),)]
    if isinstance(first, (list, tuple)):
        forms.append(tuple(first))
    if hasattr(first, "__len__"):
        forms += [(first, len(first), *args[1:]), (*args, len(first))]
    return orders[1:] + forms


def {entry_point}(*args):
    try:
        return wse_function(*args)
    except Exception as failure:
        for variant in wse_variants(args):
            try:
                return wse_function(*variant)
            except Exception:
                pass
        raise failure
"""


def adapt_calls(code: str, entry_point: str) -> str:
    """Bind the adapter to `entry_point`; recursion in `code` bypasses it."""
    return re.sub(rf"\b{re.escape(entry_point)}\b", "wse_function", code) + ADAPTER.format(
        entry_point=entry_point
    )


# Parsing ------------------------------------------------------------------------------


FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)
DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)
PYTHON_FENCES = ("", "python", "py", "python3")
ARROWS = r"==|➞|=>|->|returns|should return|gives|is"
AFFIRMATIVE_RE = re.compile(
    r"^\W*(yes|yeah|yep|correct|exactly|right|that'?s (right|correct|what i (want|intend|mean)))\b",
    re.I,
)


def top_level_functions(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def extract_code(text: str, entry_point: str) -> str | None:
    """The last parseable block that defines `entry_point`, else the last that defines anything."""
    blocks = [body for lang, body in FENCE_RE.findall(text or "") if lang.lower() in PYTHON_FENCES]
    if not blocks and "def " in (text or ""):
        blocks = [text]
    functions = [(block, top_level_functions(block)) for block in reversed(blocks)]
    for block, names in functions:
        if entry_point in names:
            return block.strip("\n")
    for block, names in functions:
        if names:
            return block.strip("\n")
    return None


def is_demo_code(node: ast.stmt, entry_point: str) -> bool:
    """Top-level statements the model adds around the function: asserts, prints, a main guard."""
    if isinstance(node, ast.Assert):
        return True
    if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
        sides = [node.test.left, *node.test.comparators]
        return any(isinstance(side, ast.Name) and side.id == "__name__" for side in sides)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        name = getattr(node.value.func, "id", None)
        return name in ("print", "input", entry_point)
    return False


def sanitize_code(code: str, task: Task) -> str:
    """Drop top-level demo code and make sure the entry point exists."""
    code = re.sub(r"^from __future__ import .*$", "", code, flags=re.M)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    demo: set[int] = set()
    for node in tree.body:
        if is_demo_code(node, task.entry_point):
            demo.update(range(node.lineno - 1, node.end_lineno))
    lines = [line for i, line in enumerate(code.splitlines()) if i not in demo]
    cleaned = "\n".join(lines).rstrip()
    functions = top_level_functions(cleaned)
    if task.entry_point not in functions:
        target = next((name for name in task.names if name in functions), None)
        if target is None and len(functions) == 1:
            target = functions[0]
        if target is not None:
            cleaned += f"\n\n\n{task.entry_point} = {target}\n"
    return cleaned


def parse_json_object(text: str) -> dict[str, Any]:
    found = re.search(r"\{.*\}", text or "", re.S)
    try:
        value = json.loads(found.group(0)) if found else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def clean_question(text: str) -> str:
    """Reduce a model reply to one question line."""
    text = (text or "").strip()
    for _lang, body in FENCE_RE.findall(text):
        text = body.strip()
        break
    lines = [line.strip().strip("`*\"' ") for line in text.splitlines() if line.strip()]
    lines = [re.sub(r"^(question|q)\s*[:.)-]\s*", "", line, flags=re.I) for line in lines]
    question = next((line for line in lines if line.endswith("?")), lines[0] if lines else "")
    if question.count("?") > 1:
        question = question[: question.index("?") + 1]
    return question[:600]


def balanced_call(line: str, start: int) -> int:
    """The index after the parenthesis that closes the one at `start`, or -1."""
    depth, quote = 0, ""
    for i in range(start, len(line)):
        ch = line[i]
        if quote:
            if ch == quote and line[i - 1] != "\\":
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def extract_examples(text: str, names: list[str], limit: int = 8) -> list[tuple[str, str]]:
    """Collect `call -> expected` pairs from doctests and inline examples."""
    examples: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(">>>"):
            call = stripped[3:].strip()
            expected = lines[i + 1].strip() if i + 1 < len(lines) else ""
            for quote in ('"""', "'''"):
                expected = expected.split(quote, 1)[0].strip()
            if call and expected and not expected.startswith(">>>"):
                examples.append((call, expected))
            i += 2 if expected else 1
            continue
        for name in names:
            for match in re.finditer(rf"\b{re.escape(name)}\(", stripped):
                end = balanced_call(stripped, match.end() - 1)
                arrow = re.match(rf"\s*({ARROWS})\s*(.+)$", stripped[end:]) if end > 0 else None
                if arrow:
                    examples.append(
                        (stripped[match.start() : end], arrow.group(2).strip().rstrip(".,;"))
                    )
        i += 1
    return list(dict.fromkeys(examples))[:limit]


def parse_calls(text: str, names: list[str], limit: int) -> list[str]:
    """Keep the lines that are one complete, parseable call of the function."""
    calls: list[str] = []
    for line in (text or "").splitlines():
        call = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip().strip("`").rstrip(",;")
        name = next((n for n in names if call.startswith(n + "(")), None)
        if name is None or balanced_call(call, len(name)) != len(call):
            continue
        try:
            ast.parse(call, mode="eval")
        except SyntaxError:
            continue
        if call not in calls:
            calls.append(call)
    return calls[:limit]


# Comparing candidates ------------------------------------------------------------------


def score(outputs: list[Outputs], expected: list[str | None], strength: list[int]) -> list[Key]:
    """Sort keys: answer checks, then task examples, then fewest errors, then agreement."""
    keys: list[Key] = []
    for i, own in enumerate(outputs):
        if own is None:
            keys.append(None)
            continue
        answer = example = errors = agreement = 0
        for j, value in enumerate(own):
            if value is None:
                continue
            if value.startswith("!"):
                errors += 1
                continue
            agreement += sum(
                o is not None and o[j] == value for k, o in enumerate(outputs) if k != i
            )
            if value == expected[j]:
                answer += strength[j] == ANSWER
                example += strength[j] == EXAMPLE
        keys.append((answer, example, -errors, agreement))
    return keys


def shown_calls(
    outputs: list[Outputs], calls: list[str], leader: int, first_probe: int
) -> list[tuple[str, str]]:
    """Up to two calls and their values for a question.

    Model-written calls with a non-trivial result come first, since the user learns little about
    the rule from an empty, zero or None case; within each group calls on which the candidates
    disagree come first. The task's own examples come last.
    """
    own = outputs[leader]
    order = []
    for j in range(first_probe, len(calls)):
        values = {o[j] for o in outputs if o is not None and o[j]}
        edge = (own[j] or "")[1:].strip() in TRIVIAL_VALUES
        order.append((edge, len(values) <= 1, j))
    ordinary = {j for edge, _agrees, j in order if not edge}
    shown: list[tuple[str, str]] = []
    for j in [j for _edge, _agrees, j in sorted(order)] + list(range(first_probe)):
        value = own[j]
        if not value or not value.startswith("=") or len(value) > 61 or len(calls[j]) > 80:
            continue
        if shown and (j not in ordinary or value[1:] == shown[0][1]):
            continue
        shown.append((calls[j], value[1:]))
        if len(shown) == 2:
            break
    return shown


# State ---------------------------------------------------------------------------------


@dataclass
class Exchange:
    """One question with its answer. A described draft only counts as far as the answer confirms."""

    question: str
    answer: str
    describes: bool
    shown: list[tuple[str, str]]
    checks: list[tuple[str, str]] | None = None  # the calls the answer settles, written once

    @property
    def informative(self) -> bool:
        return bool((self.answer or "").strip()) and not NON_ANSWER_RE.match(self.answer or "")

    @property
    def confirms(self) -> bool:
        return bool(AFFIRMATIVE_RE.match(self.answer or ""))

    def as_prompt(self, i: int) -> str:
        if not self.describes:
            return f"Q{i}: {self.question}\nA{i}: {self.answer}"
        correction = "" if self.confirms else DRAFT_NOTE_CORRECTION.format(i=i)
        return f"Q{i}{DRAFT_NOTE_QUESTION}: {self.question}\nA{i}: {self.answer}{correction}"

    def rejects(self, example: tuple[str, str]) -> bool:
        """An example the question showed with the draft's value, unless the answer confirms it."""
        squeezed = tuple("".join(part.split()) for part in example)
        shown = {tuple("".join(part.split()) for part in pair) for pair in self.shown}
        return not self.confirms and squeezed in shown


@dataclass
class Selection:
    """What a further question needs from the last selection."""

    outputs: list[Outputs]
    calls: list[str]
    chosen: int
    first_probe: int
    behaviours: int


@dataclass
class Task:
    prompt: str
    entry_point: str
    names: list[str]
    headerless: bool  # MBPP-style tasks state no header: the tests' argument list is unknown
    examples_doubtful: bool = False  # the question was about a conflict with the examples
    exchanges: list[Exchange] = field(default_factory=list)
    selection: Selection | None = None
    version: int = 0

    @classmethod
    def from_problem(cls, problem: dict[str, Any]) -> Task:
        prompt = problem["prompt"]
        entry_point = problem.get("entry_point") or "solution"
        found = DEF_RE.findall(prompt)
        return cls(
            prompt=prompt,
            entry_point=entry_point,
            names=list(dict.fromkeys([*found, entry_point])),
            headerless=entry_point not in found,
        )

    @property
    def answered(self) -> list[Exchange]:
        return [item for item in self.exchanges if item.informative]

    def clarifications(self) -> str:
        if not self.answered:
            return ""
        rows = [item.as_prompt(i) for i, item in enumerate(self.answered, 1)]
        return "\n".join([CLARIFICATIONS_HEADING, *rows]) + "\n"

    def fields(self, **extra: Any) -> dict[str, Any]:
        return {"entry_point": self.entry_point, "prompt": self.prompt, **extra}


# Algorithm -----------------------------------------------------------------------------


class WSEClarifier(ClarificationAlgorithmBase):
    # event_sink: optional callable receiving trace events; unused in an evaluation run.
    DEFAULT_CONFIG = {"event_sink": None}

    def trace(self, kind: str, **data: Any) -> None:
        sink = self.config.get("event_sink")
        if callable(sink):
            sink({"kind": kind, **data})

    def new_version(self, task: Task, kind: str, code: str) -> None:
        task.version += 1
        self.trace(kind, version=task.version, round=len(task.exchanges), code=code)

    # Model and sandbox ---------------------------------------------------------------

    def chat(self, env: ClarificationEnvironment, prompt: str) -> str:
        """One model call. A call that rich's console hook rejects is repeated once with "[/"
        defused and a hint that keeps it out of the response."""
        system = {"role": "system", "content": SYSTEM_PROMPT}
        try:
            return env.llm([system, {"role": "user", "content": prompt}]) or ""
        except LimitsExceededException:
            raise
        except Exception as error:
            if "MarkupError" not in repr(type(error)):
                raise
        defused = prompt.replace("[/", "[ /") + MARKUP_HINT
        return env.llm([system, {"role": "user", "content": defused}]) or ""

    def run_all(
        self,
        env: ClarificationEnvironment,
        task: Task,
        sources: list[str],
        calls: list[str],
        expected: list[str | None],
    ) -> tuple[list[Outputs], list[str | None]] | None:
        """Every source on every call: their outputs, and the expected values in the same form."""
        script = HARNESS.format(
            sources=repr([PRELUDE + source for source in sources]),
            calls=repr(calls),
            expected=repr(expected),
            entry_point=task.entry_point,
            aliases=repr([name for name in task.names if name != task.entry_point]),
        )
        try:
            output = env.exec_code(script) or ""
            report = json.loads(output.split(HARNESS_MARK, 1)[1].strip().splitlines()[0])
        except Exception as error:
            self.trace("error", message=repr(error)[:300])
            return None
        if len(report["candidates"]) != len(sources):
            return None
        return [c.get("outputs") for c in report["candidates"]], report["expected"]

    # Writing code --------------------------------------------------------------------

    def generate(self, env: ClarificationEnvironment, task: Task, rule: str = "") -> str | None:
        """One implementation of the task as it stands, with all answers."""
        rules = GENERATE_LENGTH_RULE if task.headerless else ""
        rules += INPUT_RULE + rule
        prompt = GENERATE_TEMPLATE.format(
            **task.fields(clarifications=task.clarifications(), signature_rule=rules)
        )
        for _ in range(WRITE_ATTEMPTS):
            code = extract_code(self.chat(env, prompt), task.entry_point)
            if code is not None:
                return sanitize_code(code, task)
        return None

    def written_calls(self, env: ClarificationEnvironment, task: Task) -> list[str]:
        """Calls the model writes to compare implementations."""
        prompt = CALLS_TEMPLATE.format(
            **task.fields(count=SELECTION_CALLS, clarifications=task.clarifications())
        )
        return parse_calls(self.chat(env, prompt), task.names, SELECTION_CALLS)

    # Questions -----------------------------------------------------------------------

    def describe(
        self,
        env: ClarificationEnvironment,
        task: Task,
        code: str,
        shown: list[tuple[str, str]],
    ) -> str | None:
        """A question that describes `code` with the shown calls and asks whether it is meant."""
        reply = self.chat(env, CONTRACT_TEMPLATE.format(**task.fields(code=code)))
        args, returns, rule = (str(parse_json_object(reply).get(k) or "").strip() for k in CONTRACT)
        if not (args and returns and rule):
            return None
        rule = rule.rstrip(".") + "."
        facts = task.fields(
            args=args.rstrip("."), returns=returns.rstrip("."), rule=rule[0].lower() + rule[1:]
        )
        example = ", and ".join(f"{call} returns {value}" for call, value in shown)
        return CONTRACT_QUESTION.format(
            **facts, example=f" For example, {example}." if shown else ""
        )

    def first_question(
        self, env: ClarificationEnvironment, task: Task, code: str
    ) -> tuple[str | None, str | None, list[tuple[str, str]]]:
        """The question, the draft it describes (if any) and the calls it shows."""
        # Repeated identical prompts return nearly the same draft, so every further draft reads
        # the task under another instruction.
        drafts = [code]
        for framing in DRAFT_FRAMINGS:
            draft = self.generate(env, task, framing)
            if draft is not None:
                drafts.append(draft)
        stated = extract_examples(task.prompt, task.names)
        calls = [call for call, _value in stated]
        calls += [call for call in self.written_calls(env, task) if call not in calls]
        expected = [value for _call, value in stated] + [None] * (len(calls) - len(stated))
        strength = [EXAMPLE] * len(stated) + [PROBE] * (len(calls) - len(stated))
        result = self.run_all(env, task, drafts, calls, expected) if calls else None
        keys = score(*result, strength) if result else []
        ranked = [i for i, key in enumerate(keys) if key is not None]
        if ranked:
            leader = max(ranked, key=lambda i: keys[i])
            if stated and keys[leader][1] * 2 < len(stated):
                task.examples_doubtful = True
                return self.gap_question(env, task, EXAMPLES_CONFLICT), None, []
            shown = shown_calls(result[0], calls, leader, len(stated))
            question = self.describe(env, task, drafts[leader], shown)
            if question:
                return question, drafts[leader], shown
        return self.gap_question(env, task, TEXT_GAP), None, []

    def gap_question(self, env: ClarificationEnvironment, task: Task, situation: str) -> str:
        return clean_question(
            self.chat(env, GAP_TEMPLATE.format(**task.fields(situation=situation)))
        )

    # Selection -----------------------------------------------------------------------

    def checks(
        self, env: ClarificationEnvironment, task: Task
    ) -> tuple[list[str], list[str | None], list[int]]:
        """Calls with their expected values and strength: answers first, then the task's examples."""
        answers = "\n".join(item.answer for item in task.answered)
        pairs = [(e, ANSWER) for e in extract_examples(answers, task.names)]
        for item in task.answered:
            if item.checks is None:
                reply = self.chat(
                    env,
                    ANSWER_CHECKS_TEMPLATE.format(
                        **task.fields(question=item.question, answer=item.answer)
                    ),
                )
                found = extract_examples(reply, task.names, limit=5)
                item.checks = [e for e in found if not item.rejects(e)]
            pairs += [(e, ANSWER) for e in item.checks]
        if not task.examples_doubtful:
            pairs += [(e, EXAMPLE) for e in extract_examples(task.prompt, task.names)]
        table: dict[str, tuple[str, int]] = {}
        for (call, value), strength in pairs:
            table.setdefault(call, (value, strength))
        return list(table), [v for v, _s in table.values()], [s for _v, s in table.values()]

    def select(self, env: ClarificationEnvironment, task: Task, code: str, draft: str) -> str:
        """Write more implementations and return the one that best fits the checks. The earlier
        implementation takes part in the comparison but cannot be chosen."""
        written = [code]
        for _ in range(SELECTION_CANDIDATES - 1):
            extra = self.generate(env, task)
            if extra is not None:
                written.append(extra)
        candidates = written + [draft]
        for i, candidate in enumerate(candidates):
            self.trace("candidate", candidate=i, code=candidate)
        calls, expected, strength = self.checks(env, task)
        stated = len(calls)
        for call in self.written_calls(env, task):
            if call not in calls:
                calls.append(call)
                expected.append(None)
                strength.append(PROBE)
        result = self.run_all(env, task, candidates, calls, expected) if calls else None
        if result is None:
            return code
        outputs, expectations = result
        keys = score(outputs, expectations, strength)
        loaded = [i for i in range(len(written)) if keys[i] is not None]
        if not loaded:
            return code
        best = max(loaded, key=lambda i: keys[i])
        task.selection = Selection(
            outputs=outputs[: len(written)],
            calls=calls,
            chosen=best,
            first_probe=stated,
            behaviours=len({tuple(o) for o in outputs if o is not None}),
        )
        self.new_version(task, "selection", written[best])
        return written[best]

    # Orchestration -------------------------------------------------------------------

    def finish(self, task: Task, code: str | None) -> str:
        if code is None:
            return f"```python\ndef {task.entry_point}(*args):\n    raise NotImplementedError\n```"
        code = PRELUDE + adapt_calls(code, task.entry_point)
        lower = task.entry_point.lower()
        if lower != task.entry_point:
            code += f"\n\n{lower} = {task.entry_point}\n"
        return f"```python\n{code}\n```"

    def ask(self, env: ClarificationEnvironment, task: Task, code: str) -> str | None:
        """Ask the next question if one is worth its discount, and return the draft it concerns.
        Returns None when no question was asked."""
        if not task.exchanges:
            question, draft, shown = self.first_question(env, task, code)
            describes = draft is not None
            draft = draft or code
        else:
            last = task.selection
            if last is None or (task.exchanges[-1].informative and last.behaviours < 2):
                return None
            shown = shown_calls(last.outputs, last.calls, last.chosen, last.first_probe)
            question = self.describe(env, task, code, shown)
            describes, draft = True, code
        if not question:
            return None
        # The official parser reads the answer up to the first backtick, and the official runner
        # prints the question through rich before asking.
        question = " ".join(question.replace("`", "").split()).replace("[/", "[ /")
        if question.strip().lower() in {item.question.strip().lower() for item in task.exchanges}:
            return None
        try:
            answer = env.ask_human(question)
        except TooManyQuestionException:
            return None
        task.exchanges.append(Exchange(question, answer, describes, shown))
        return draft

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        task = Task.from_problem(problem)
        code: str | None = None
        try:
            code = self.generate(env, task)
            if code is None:
                return self.finish(task, None)
            self.new_version(task, "generate", code)
            while env.can_ask():
                draft = self.ask(env, task, code)
                if draft is None:
                    break
                updated = self.generate(env, task)
                if updated is not None:
                    code = updated
                    self.new_version(task, "update", code)
                code = self.select(env, task, code, draft)
        except LimitsExceededException:
            pass
        except Exception as error:
            # Keep the best implementation found so far instead of losing the task.
            self.trace("error", message=repr(error)[:500])
        return self.finish(task, code)
