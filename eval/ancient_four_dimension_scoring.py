#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四维释读解释评分器：一致性 / 正确性 / 具体性 / 类人性，1–4 分。

输入为字形图片与模型输出，输出每个维度的分数与评分理由。
"""

import argparse
import base64
import csv
import json
import mimetypes
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from openai import (
        APIStatusError as OpenAIStatusError,
        OpenAI,
        RateLimitError,
    )
except ImportError:  # pragma: no cover - urllib fallback keeps the scorer runnable.
    OpenAI = None  # type: ignore[assignment]

    class OpenAIStatusError(Exception):
        pass

    class RateLimitError(Exception):
        pass


if sys.stdout.encoding and sys.stdout.encoding.lower() in {"gbk", "cp936", "cp950"}:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 唯一口径常量
# ---------------------------------------------------------------------------

SCORE_MIN = 1
SCORE_MAX = 4
SCORE_SCALE_LABEL = "1-4 (0.5 steps)"

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
APIKEY = ""
DEFAULT_JUDGE_MODEL = "deepseek-v4.1-flash"

ANCIENT_ANSWER_FIELDS = ["output", "explanation", "answer"]

DIMENSIONS = [
    "consistency",
    "correctness",
    "specificity",
    "human_likeness",
]

DIMENSION_NAMES_ZH = {
    "consistency": "一致性",
    "correctness": "正确性",
    "specificity": "具体性",
    "human_likeness": "类人性",
}

SCRIPT_ORACLE_BONE = "甲骨文"
SCRIPT_HOUMA = "侯马盟书"
DEFAULT_SCRIPT = SCRIPT_ORACLE_BONE

SCRIPT_ALIASES: List[Tuple[str, Tuple[str, ...]]] = [
    (SCRIPT_HOUMA, ("侯马", "侯馬", "houma")),
    (SCRIPT_ORACLE_BONE, ("甲骨文", "甲骨", "oracle", "evobc", "obc", "obi")),
]

SCRIPT_DETECT_FIELDS = (
    "id",
    "obi_test_id",
    "task",
    "source",
    "image",
    "images",
    "instruction",
)

SCRIPT_NAMES_EN = {
    SCRIPT_ORACLE_BONE: "oracle-bone script",
    SCRIPT_HOUMA: "Houma covenant script",
}

SCRIPT_NAMES_ZH_RUBRIC = {
    SCRIPT_ORACLE_BONE: "甲骨文",
    SCRIPT_HOUMA: "侯马盟书",
}

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1200

DEFAULT_RATE_LIMIT_BASE_SLEEP = 30.0
DEFAULT_RATE_LIMIT_MAX_SLEEP = 300.0
DEFAULT_EMPTY_RESPONSE_BASE_SLEEP = 5.0
DEFAULT_EMPTY_RESPONSE_MAX_SLEEP = 30.0

VISIBLE_JSON_INSTRUCTION = (
    "Important response requirement: Return a visible JSON object in the response content. "
    "Do not put the answer only in hidden reasoning. Do not return markdown or extra text."
)

# 关闭判官思考模式，避免 JSON 落入 reasoning_content。
DISABLE_THINKING = False
THINKING_EFFORT = ""

SCRIPT_PLACEHOLDER = "{script}"
SCHEMA_PLACEHOLDER = "{schema}"


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------

# 单份 system prompt：任务 → 判定原则 → 四维锚点 → 判分方式 → 评分规则 → JSON。
SYSTEM_PROMPT_TEMPLATE = """你是{script}释读质量的严格评审者。请对给定模型的推理文本按四个维度打分，每维 1-4 分。不要改写答案。

【任务】
- 评分对象是模型的完整输出。
- 你只会看到字形图片和这段输出：不给参考答案，也不给题干。
- 只看展示出来的推理。只报出字名、不说明任何形体依据的，属于缺少构件分析：具体性判低（1-2 分）。论断少，不等于有矛盾。

【判定原则】（适用所有维度）
- 只评古文字学内容，不评呈现形式。篇幅长短、是否分点、语气自信或犹疑，本身不加分也不减分。
- 自信不是证据。一个读法是否成立，看它援引的构件与结论之间是否对得上。如实指出不确定，是准确，不是缺点。
- 评分对象是这段解释自身的论证结构：它怎样把构件对应到字。不要因为某个部件名是现代通行写法就判它错误；
  同一构件在不同书体里写法不同，解释用现代名称指代它并不构成错误。
- 逐项核对解释声称可见的笔画、构件及其位置是否能在图片中辨认；不能只因解释自洽或语气确定就接受其判断。
- 正确性评估字形证据是否支持解释提出的构件识别及其组合。不要用通行释读替代图像核验；但若构件与图片明显不符，应据此扣分。
- 解释承认某些笔画模糊、只能作相对判断，属于恰当处理，不扣分。
- 由字形的整体形似或构形特征出发、再落实到具体释读，是古文字学的正当路径（象形、会意字的
  释读常由此入手）。走这条路径、并说明了据以判断的形体特征的，与逐构件推导同等有效：
  不得因为「没有把每个构件逐一推导到结论」而判为缺环、留下未说明环节，或据此扣分。
  只要解释说清了「看到什么形体特征」与「由此得出哪个字」的对应，就按给出了依据处理。
- 被检验的是证据链是否完整、结论能否由前面的陈述推出；先写哪一部分、是否分点，本身不影响分数。
- 不要因为解释逐个比较候选字「哪里不像」「与选项不符」而加分。从给定的候选字反推答案，
  是排除法猜题，不是从字形自主释读：这段内容既不能替代构件证据，也不提高任何维度的分数。
- 出现候选字逐一排除（如「A 应为…与图不符」「B 左边是…）的段落时，按篇幅占比判类人性与正确性：
  占三成以下，类人性至多 3.5；占三到六成，类人性至多 3；占六成以上，正确性与类人性至多 2.5。
  该段落的篇幅按它占整段解释的句子数估算。
  反过来，完全不提候选字、只凭字形自身特征得出结论的，属于正当的自主释读，不因此扣分。

【四个维度】
（每维 1-4 分，可用 0.5 档；1 = 不合格，4 = 完整合格。4 分描述的是合格线，不是专家线。）

1. 一致性（consistency）：评估推理是否锚定图中字形、前后是否自洽。
   1 = 说法前后自相矛盾，或与所给字形的整体形态明显抵触。
   2 = 论证方向来回摇摆，或多处与前文脱节。
   3 = 能围绕该字形说下来，个别处前后衔接不紧。
   4 = 围绕该字形展开，前后自洽，没有自相矛盾的陈述。
   注意以下都不算自相矛盾，不得据此扣一致性分：
   - 先列出几种可能读法、随后收敛到一个结论（这是正常的推定过程）；
   - 对某构件先作试探性命名、后文改用更准确的说法；
   - 如实说明某处不确定、只能作相对判断；
   - 较长解释中先交代整体形态、后逐部分展开。

2. 正确性（correctness）：评估构件分析在古文字学上是否说得通。
   1 = 构件拆分不能成立，或与最终所得的字直接冲突。
   2 = 有一个主要构件完全没有交代，或某个构件与最终结论互相抵触。
   3 = 每个主要构件都有交代，且组合起来能推出所主张的字；个别措辞含混不影响。
   4 = 每个主要构件的对应都写清楚了，从构件到结论没有留下未说明的环节。

3. 具体性（specificity）：评估解释是否针对本字形给出可核对的构件、方位与布局。
   1 = 只有字名或结论，没有任何关于构件的描述。
   2 = 只给出部件名，完全没有任何方位或形态说明。
   3 = 提到了部分构件的方位或形态。
   4 = 主要构件的方位与形态大体都交代了，读者可在图上对照。

4. 类人性（human_likeness）：评估解释是否体现古文字学者实际读字的方式而非常识性套话。
   1 = 只给结论，没有任何拆字过程。
   2 = 完全套用字典拆字，对字形本身只字未提。
   3 = 提到了字形本身的特征，按字形说明的尝试可见。
   4 = 按字形逐部分说明构件，读字方式符合本学科习惯。

【判分方式】
- 四维各自独立评估，每个维度只按它自己的定义给分。
- 在 rationale 中按 consistency / correctness / specificity / human_likeness 的顺序，各写一句该维度自己的判断依据。
- 四个维度必须分别判断，但不同维度可以合理得到相同分数；不要为了制造分差而调整任一维度的分数。

【评分规则】
- 每个维度只能给 1、1.5、2、2.5、3、3.5、4 之一，可以带 .5，不给其他小数。
- 4 分是「合格完整」，不是「专家级」：部件齐全、大致落到笔画、推导完整、无明确错误，即为 4 分；
  不需要给出排除竞争读法的论证。
- 3.5 = 达到 4 分且没有留下可疑环节；3 = 达到合格但有明确短板；
  2.5 = 介于合格与不合格之间；2 = 有明显缺口；1.5 = 接近不合格；1 = 不合格。
- 不要因为「不够专家级」而降档，只在确有缺陷时降档。
- 扣分必须能指出具体缺陷：哪个主要构件完全没交代、哪句话自相矛盾、哪个构件与结论直接抵触。
  说不出这类具体缺陷时给 3 分或以上。以下都不是扣分理由：衔接不够紧、对应不够明确、
  表述较笼统、留有未说明的环节、未逐笔落到具体构件、深度有限、有套用字典拆字的痕迹。
- 3 分是正常解释的默认档。2.5 只用于「可以明确指认出缺了哪个主要构件」的情形，
  不要把它当作保守的中间默认值。
- 当解释做到了以下几点时给 3.5：主要构件逐一说清、每个构件都能看出在说图上的哪一部分、
  从构件到结论没有需要读者自行补全的地方。做到这些又通篇没有可指认的缺陷时给 4。
  「写得可以更好」不是不给 3.5/4 的理由。
- 评分分布参考：同一批输出里，多数完整作答的解释应落在 3.5 到 4 之间；3.5 和 4 合计约六成，
  3 约三成，2.5 及以下约一成。如果你的结果大半停在 3 分，说明你仍把「本可写得更细」
  当成了缺陷。
- 「某个构件交代得笼统、没有逐笔落到图上」只在整段解释确实缺了这一环时才算缺陷；
  如果解释已经说出了该构件的方位或形态，即便不够精确，也按 3.5 计，不按 3 计。

{schema}"""


JSON_SCHEMA = """你必须以如下 JSON 格式输出，且 JSON 必须出现在可见回复正文中
（不要只放在隐藏推理里），不要输出 markdown 或多余文字：
{
  "scores": {
    "consistency": 1 到 4，可用 0.5 档（如 2.5）,
    "correctness": 1 到 4，可用 0.5 档（如 2.5）,
    "specificity": 1 到 4，可用 0.5 档（如 2.5）,
    "human_likeness": 1 到 4，可用 0.5 档（如 2.5）
  },
  "rationale": "按四个维度各写一句打分依据"
}"""


def build_system_prompt(script: str = DEFAULT_SCRIPT) -> str:
    """填字体名与 JSON schema，得到判官看到的完整 system message。"""
    script_name = SCRIPT_NAMES_ZH_RUBRIC.get(script, script)
    return (
        SYSTEM_PROMPT_TEMPLATE
        .replace(SCRIPT_PLACEHOLDER, script_name)
        .replace(SCHEMA_PLACEHOLDER, JSON_SCHEMA)
    )


# ---------------------------------------------------------------------------
# 字体判定
# ---------------------------------------------------------------------------

def normalize_script(value: str) -> str:
    haystack = str(value or "").strip().lower()
    if not haystack:
        return ""
    for name, aliases in SCRIPT_ALIASES:
        if any(alias.lower() in haystack for alias in aliases):
            return name
    raise ValueError(
        f"Unknown script '{value}'. Expected one of: "
        + ", ".join(name for name, _ in SCRIPT_ALIASES)
        + " (ASCII aliases such as 'houma' / 'oracle' also work)."
    )


def detect_script(row: Dict[str, Any], forced: str = "") -> str:
    if forced:
        return normalize_script(forced)
    haystack = " ".join(
        str(row.get(field, "") or "") for field in SCRIPT_DETECT_FIELDS
    ).lower()
    for name, aliases in SCRIPT_ALIASES:
        if any(alias.lower() in haystack for alias in aliases):
            return name
    return DEFAULT_SCRIPT


# ---------------------------------------------------------------------------
# 传输层
# ---------------------------------------------------------------------------

_CLIENT_CACHE: Dict[Tuple[str, str, int], Any] = {}
_CLIENT_LOCK = threading.Lock()


def normalize_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def get_client(api_key: str, base_url: str, timeout: int) -> Any:
    key = (api_key, normalize_base_url(base_url), int(timeout))
    with _CLIENT_LOCK:
        client = _CLIENT_CACHE.get(key)
        if client is None:
            if OpenAI is None:
                client = {
                    "api_key": api_key,
                    "base_url": normalize_base_url(base_url),
                    "timeout": int(timeout),
                }
                _CLIENT_CACHE[key] = client
                return client
            http_client = None
            try:
                import httpx

                http_client = httpx.Client(trust_env=False, timeout=float(timeout))
            except Exception:
                http_client = None
            client = OpenAI(
                api_key=api_key,
                base_url=normalize_base_url(base_url),
                timeout=float(timeout),
                max_retries=0,
                **({"http_client": http_client} if http_client is not None else {}),
            )
            _CLIENT_CACHE[key] = client
        return client


def compact_json(data: Any, max_len: int = 1500) -> str:
    text = json.dumps(data, ensure_ascii=False)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def post_json(url: str, api_key: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"content-type": "application/json", "accept": "application/json",
                 "user-agent": "Mozilla/5.0", "authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_text = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        raw_text = e.read().decode("utf-8", errors="replace")
        status = e.code
    try:
        data = json.loads(raw_text)
    except Exception:
        data = {"raw_text": raw_text[:3000]}
    if status == 429:
        raise RateLimitError(f"HTTP 429: {compact_json(data)}")
    if status >= 400:
        raise OpenAIStatusError(f"HTTP {status}: {compact_json(data)}")
    return data


def post_chat_stream(url: str, api_key: str, payload: Dict[str, Any], timeout: int) -> str:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"content-type": "application/json", "accept": "text/event-stream",
                 "user-agent": "Mozilla/5.0", "authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raw_text = e.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw_text)
        except Exception:
            data = {"raw_text": raw_text[:3000]}
        if e.code == 429:
            raise RateLimitError(f"HTTP 429: {compact_json(data)}")
        raise OpenAIStatusError(f"HTTP {e.code}: {compact_json(data)}") from e
    parts: List[str] = []
    with resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                parts.append(str(content))
    return "".join(parts)


def extract_chat_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


def create_completion(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int,
    use_stream: bool,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    extra: Dict[str, Any] = {}
    if temperature >= 0:
        extra["temperature"] = temperature
    if max_tokens > 0:
        extra["max_tokens"] = max_tokens
    if DISABLE_THINKING:
        # 必须走 extra_body：OpenAI SDK 只接受已知的顶层参数。
        extra["extra_body"] = {"enable_thinking": False}
    elif THINKING_EFFORT:
        extra["extra_body"] = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": THINKING_EFFORT,
        }
    # deepseek-flash 默认开思考，JSON 会落进 reasoning_content、可见 content 为空。
    elif str(model).startswith("deepseek"):
        extra["extra_body"] = {"thinking": {"type": "disabled"}}

    if isinstance(client, dict):
        payload: Dict[str, Any] = {"model": model, "messages": messages, **extra}
        url = f"{client['base_url']}/chat/completions"
        if use_stream:
            payload["stream"] = True
            return post_chat_stream(url, client["api_key"], payload, timeout)
        data = post_json(url, client["api_key"], payload, timeout)
        return extract_chat_text(data)

    if use_stream:
        stream = client.chat.completions.create(
            model=model, messages=messages, stream=True, timeout=timeout, **extra,
        )
        parts: List[str] = []
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta else None
            if content:
                parts.append(content)
        return "".join(parts)

    resp = client.chat.completions.create(
        model=model, messages=messages, timeout=timeout, **extra,
    )
    choices = getattr(resp, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message else None
    return content or ""


def retry_after_from_error(error: Exception) -> Optional[float]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    return parse_retry_after(headers.get("retry-after") or headers.get("Retry-After"))


def parse_retry_after(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except Exception:
        return None
    if seconds < 0:
        return None
    return seconds


class EmptyJudgeResponseError(RuntimeError):
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        super().__init__(f"Empty judge response text: {compact_json(data)}")


def build_judge_input(content: Any, attempt: int = 0) -> Any:
    retry_note = ""
    if attempt > 0:
        retry_note = (
            f"\n\nRetry {attempt}: the previous reply did not contain a visible JSON object "
            "matching the schema. Output strictly in the JSON format above, with all four "
            "dimensions present, each a number 1-4 in 0.5 steps."
        )
    instruction = f"{VISIBLE_JSON_INSTRUCTION}{retry_note}"
    if isinstance(content, list):
        return [{"type": "text", "text": instruction}] + content
    return f"{instruction}\n\n{content}"


def call_judge(
    api_key: str,
    base_url: str,
    model: str,
    content: Any,
    timeout: int,
    retries: int,
    use_stream: bool = True,
    rate_limit_base_sleep: float = DEFAULT_RATE_LIMIT_BASE_SLEEP,
    rate_limit_max_sleep: float = DEFAULT_RATE_LIMIT_MAX_SLEEP,
    empty_response_base_sleep: float = DEFAULT_EMPTY_RESPONSE_BASE_SLEEP,
    empty_response_max_sleep: float = DEFAULT_EMPTY_RESPONSE_MAX_SLEEP,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    parse: Optional[Callable[[str], Any]] = None,
    system_prompt: str = "",
) -> Any:
    client = get_client(api_key, base_url, timeout)
    system = system_prompt or build_system_prompt(DEFAULT_SCRIPT)
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": build_judge_input(content, attempt=attempt)},
        ]
        try:
            text = create_completion(
                client, model, messages, timeout, use_stream, temperature, max_tokens
            )
            if not text.strip():
                raise EmptyJudgeResponseError({"note": "empty assistant content"})
            return text if parse is None else parse(text)
        except Exception as e:
            last_error = e
            if attempt >= retries:
                break
            if isinstance(e, RateLimitError):
                base_sleep = retry_after_from_error(e)
                if base_sleep is None:
                    base_sleep = min(rate_limit_base_sleep * (2 ** attempt), rate_limit_max_sleep)
                else:
                    base_sleep = min(base_sleep, rate_limit_max_sleep)
                time.sleep(base_sleep + random.random())
            elif isinstance(e, EmptyJudgeResponseError):
                base_sleep = min(empty_response_base_sleep * (2 ** attempt), empty_response_max_sleep)
                time.sleep(base_sleep + random.random())
            else:
                time.sleep(min(3 * (2 ** attempt), 60) + random.random())
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


# ---------------------------------------------------------------------------
# 解析与校验
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    raise ValueError(f"Cannot parse judge JSON: {text[:500]}")


def normalize_scores(data: Dict[str, Any]) -> Dict[str, float]:
    """读取四个维度分并强制 4 点量表（允许 .5 档）。

    缺维度、非数字、非 0.5 的倍数、越界一律抛错 → call_judge 重试，而不是静默截断。
    """
    raw_scores = data.get("scores", data)
    if not isinstance(raw_scores, dict):
        raise ValueError(f"Judge JSON has no usable 'scores' object: {compact_json(data)}")
    scores: Dict[str, float] = {}
    for name in DIMENSIONS:
        if name not in raw_scores:
            raise ValueError(f"Judge JSON is missing dimension '{name}': {compact_json(data)}")
        try:
            value = float(raw_scores[name])
        except (TypeError, ValueError):
            raise ValueError(
                f"Judge JSON dimension '{name}' is not a number: {raw_scores[name]!r}"
            )
        if abs(value * 2 - round(value * 2)) > 1e-9:
            raise ValueError(f"{name} score is not a multiple of 0.5: {value}")
        if value < SCORE_MIN - 1e-9 or value > SCORE_MAX + 1e-9:
            raise ValueError(f"{name} score out of range {SCORE_MIN}-{SCORE_MAX}: {value}")
        scores[name] = round(value, 1)
    return scores


# ---------------------------------------------------------------------------
# 图片
# ---------------------------------------------------------------------------

_IMAGE_FIELDS = ("image", "images", "img", "image_path", "image_file")


def resolve_image_ref(sample: Dict[str, Any]) -> str:
    for field in _IMAGE_FIELDS:
        value = sample.get(field)
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        if isinstance(value, str) and value.strip():
            return value.strip()
    rid = str(sample.get("id", ""))
    if "|" in rid:
        return rid.rsplit("|", 1)[1].strip()
    return ""


def image_data_url(path: str, root: str = "") -> str:
    if not path:
        return ""
    norm = path.replace("\\", "/")
    candidates: List[str] = []
    if os.path.isabs(norm):
        candidates.append(norm)
    else:
        if root:
            candidates.append(os.path.join(root, norm))
        candidates.append(norm)
        if not norm.startswith("data/"):
            candidates.append(os.path.join("data", norm))
    for candidate in candidates:
        if os.path.isfile(candidate):
            mime = mimetypes.guess_type(candidate)[0] or "image/png"
            with open(candidate, "rb") as fh:
                return f"data:{mime};base64,{base64.b64encode(fh.read()).decode('ascii')}"
    return ""


# ---------------------------------------------------------------------------
# 单条打分
# ---------------------------------------------------------------------------

def build_user_text(sample: Dict[str, Any], script: str = DEFAULT_SCRIPT) -> str:
    return (
        f"Evaluate the four-dimension quality of the model's decipherment reasoning for "
        f"the {SCRIPT_NAMES_EN.get(script, script)} glyph shown in the image.\n\n"
        f"Model reasoning path:\n{sample.get('output') or ''}\n"
    )


def score_explanation(
    sample: Dict[str, Any],
    script: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    retries: int,
    use_stream: bool,
    rate_limit_base_sleep: float,
    rate_limit_max_sleep: float,
    empty_response_base_sleep: float,
    empty_response_max_sleep: float,
    temperature: float,
    max_tokens: int,
    image_root: str = "",
) -> Dict[str, Any]:
    text = build_user_text(sample, script)
    system_prompt = build_system_prompt(script)

    image_ref = resolve_image_ref(sample)
    data_url = image_data_url(image_ref, image_root)
    if data_url:
        content: Any = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    else:
        content = text

    def parse(raw_text: str) -> Dict[str, Any]:
        obj = extract_json_object(raw_text)
        return {
            "scores": normalize_scores(obj),
            "rationale": str(obj.get("rationale", "")),
            "raw_judge": obj,
        }

    parsed = call_judge(
        api_key=api_key,
        base_url=base_url,
        model=model,
        content=content,
        timeout=timeout,
        retries=retries,
        use_stream=use_stream,
        rate_limit_base_sleep=rate_limit_base_sleep,
        rate_limit_max_sleep=rate_limit_max_sleep,
        empty_response_base_sleep=empty_response_base_sleep,
        empty_response_max_sleep=empty_response_max_sleep,
        temperature=temperature,
        max_tokens=max_tokens,
        parse=parse,
        system_prompt=system_prompt,
    )
    return {
        "scores": parsed["scores"],
        "rationale": parsed["rationale"],
        "judge_model": model,
        "script": script,
        "image_ref": image_ref,
        "image_used": bool(data_url),
        "raw_judge": parsed["raw_judge"],
    }


# ---------------------------------------------------------------------------
# IO 与汇总
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def get_rows(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and isinstance(obj.get("data1"), list):
        return obj["data1"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Input JSON must be a list or {'data1': list}.")


def detect_answer_field(row: Dict[str, Any], forced_field: str = "") -> str:
    if forced_field:
        if forced_field in row:
            return forced_field
        raise ValueError(f"Forced answer field '{forced_field}' not found.")
    for key in ANCIENT_ANSWER_FIELDS:
        if str(row.get(key, "")).strip():
            return key
    raise ValueError(
        "Cannot find answer field. Expected one of: " + ", ".join(ANCIENT_ANSWER_FIELDS)
    )


def item_key(row: Dict[str, Any], idx: int) -> str:
    for key in ["id", "obi_test_id", "instruction"]:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return f"idx_{idx}"


def mean_by_dimension(results: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    per_dimension: Dict[str, List[float]] = {name: [] for name in DIMENSIONS}
    for result in results:
        for name in DIMENSIONS:
            value = result["scores"].get(name, -1)
            if isinstance(value, (int, float)) and value >= 0:
                per_dimension[name].append(float(value))
    return {
        name: (round(statistics.mean(values), 4) if values else None)
        for name, values in per_dimension.items()
    }


def summarize(rows: List[Dict[str, Any]], source: str, judge_model: str) -> Dict[str, Any]:
    scored_rows = [
        row for row in rows
        if isinstance(row.get("four_dimension"), dict)
        and isinstance(row["four_dimension"].get("scores"), dict)
    ]
    failed = [row for row in rows if not isinstance(row.get("four_dimension"), dict)]
    avg_dimensions = mean_by_dimension([row["four_dimension"] for row in scored_rows])

    script_counts: Dict[str, int] = {}
    results_by_script: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        script = str(row.get("four_dimension_script") or DEFAULT_SCRIPT)
        script_counts[script] = script_counts.get(script, 0) + 1
    for row in scored_rows:
        script = str(row.get("four_dimension_script") or DEFAULT_SCRIPT)
        results_by_script.setdefault(script, []).append(row["four_dimension"])

    return {
        "source": source,
        "judge_model": judge_model,
        "total": len(rows),
        "scored": len(scored_rows),
        "failed": len(failed),
        "score_scale": SCORE_SCALE_LABEL,
        "prompt_variant": "four_dimension",
        "avg_dimensions": avg_dimensions,
        "avg_dimensions_zh": {
            DIMENSION_NAMES_ZH[name]: value for name, value in avg_dimensions.items()
        },
        "script_counts": script_counts,
        "avg_dimensions_by_script": {
            script: mean_by_dimension(results)
            for script, results in sorted(results_by_script.items())
        },
    }


def flatten_summary(data: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            items.extend(flatten_summary(value, prefix=f"{name}."))
        else:
            items.append((name, value))
    return items


def write_summary_csv(path: str, summary: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for key, value in flatten_summary(summary):
            writer.writerow([key, value])


def format_progress_bar(completed: int, total: int, width: int = 18) -> str:
    if total <= 0:
        return f"[{'-' * width}] 0/0 (100.0%)"
    completed = max(0, min(completed, total))
    filled = int(width * completed / total)
    return f"[{'#' * filled}{'-' * (width - filled)}] {completed}/{total} ({completed / total * 100:.1f}%)"


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def score_row(
    task: Tuple[int, Dict[str, Any], str],
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    idx, sample, source = task
    result = score_explanation(
        sample=sample,
        script=str(sample.get("four_dimension_script") or DEFAULT_SCRIPT),
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.judge_model,
        timeout=args.timeout,
        retries=args.retries,
        use_stream=not args.no_stream,
        rate_limit_base_sleep=args.rate_limit_base_sleep,
        rate_limit_max_sleep=args.rate_limit_max_sleep,
        empty_response_base_sleep=args.empty_response_base_sleep,
        empty_response_max_sleep=args.empty_response_max_sleep,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        image_root=args.image_root,
    )
    return idx, result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ancient-script four-dimension (1-4 Likert) explanation scorer, "
                    "four-dimension rubric prompt."
    )
    parser.add_argument("--input", default="", help="Input JSON: a list or {'data1': list}.")
    parser.add_argument("--output", default="", help="Output JSON with a four_dimension field per row.")
    parser.add_argument("--output-jsonl", default="", help="Optional JSONL copy, one line per scored row.")
    parser.add_argument("--answer-field", default="", help="Optional forced answer field.")
    parser.add_argument(
        "--image-root", default="",
        help="Base directory for resolving relative glyph-image paths. Default: cwd.",
    )
    parser.add_argument(
        "--script", default="",
        help=f"Force one script for every row ({SCRIPT_ORACLE_BONE} / {SCRIPT_HOUMA}; "
             f"aliases 'houma' / 'oracle' work). Default: auto-detect per row.",
    )
    parser.add_argument("--api-key", default=APIKEY)
    parser.add_argument("--base-url", default=os.getenv("JUDGE_BASE_URL", DEFAULT_BASE_URL).strip())
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Use non-streaming requests. Default: stream (relay's most reliable path).",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--rate-limit-base-sleep", type=float, default=DEFAULT_RATE_LIMIT_BASE_SLEEP)
    parser.add_argument("--rate-limit-max-sleep", type=float, default=DEFAULT_RATE_LIMIT_MAX_SLEEP)
    parser.add_argument("--empty-response-base-sleep", type=float, default=DEFAULT_EMPTY_RESPONSE_BASE_SLEEP)
    parser.add_argument("--empty-response-max-sleep", type=float, default=DEFAULT_EMPTY_RESPONSE_MAX_SLEEP)
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="Pass a negative value to omit the parameter for models that reject it.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Pass 0 to omit the parameter for models that reject it.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--no-think", action="store_true",
        help="Send enable_thinking=false to the judge (required by some OpenAI-compatible endpoints).",
    )
    parser.add_argument(
        "--thinking-effort", choices=["low", "high", "max"], default="",
        help="Enable model reasoning at the requested effort (DeepSeek V4 API).",
    )
    parser.add_argument(
        "--print-prompt", action="store_true",
        help="Print the full judge system message and exit (no API calls).",
    )
    args = parser.parse_args()

    global DISABLE_THINKING, THINKING_EFFORT
    DISABLE_THINKING = args.no_think
    THINKING_EFFORT = args.thinking_effort
    if DISABLE_THINKING and THINKING_EFFORT:
        raise SystemExit("--no-think and --thinking-effort cannot be used together.")

    if args.print_prompt:
        print(build_system_prompt(normalize_script(args.script) if args.script else DEFAULT_SCRIPT))
        return

    if not args.input or not args.output:
        raise SystemExit("--input and --output are required (unless --print-prompt).")

    if not args.api_key:
        raise SystemExit("Please set JUDGE_API_KEY or pass --api-key.")

    if args.script:
        try:
            args.script = normalize_script(args.script)
        except ValueError as e:
            raise SystemExit(str(e))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    source_rows = get_rows(load_json(args.input))
    if args.limit > 0:
        source_rows = source_rows[:args.limit]

    done: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(args.output):
        try:
            for idx, row in enumerate(get_rows(load_json(args.output))):
                if isinstance(row.get("four_dimension"), dict) and row["four_dimension"].get("scores"):
                    done[item_key(row, idx)] = row["four_dimension"]
        except Exception as e:
            print(f"[WARN] cannot read existing output for resume ({e}); starting fresh.", flush=True)

    rows: List[Dict[str, Any]] = []
    tasks: List[Tuple[int, Dict[str, Any], str]] = []
    failed: List[Tuple[int, str]] = []
    for idx, row in enumerate(source_rows):
        out_row = dict(row)
        key = item_key(out_row, idx)
        out_row["four_dimension_script"] = detect_script(out_row, args.script)
        if key in done:
            out_row["four_dimension"] = done[key]
            rows.append(out_row)
            continue
        try:
            answer_field = detect_answer_field(out_row, args.answer_field)
        except ValueError as e:
            out_row["four_dimension_errors"] = [str(e)]
            failed.append((idx, str(e)))
            rows.append(out_row)
            continue
        answer = str(out_row.get(answer_field, "")).strip()
        if not answer:
            message = f"Answer field '{answer_field}' is empty."
            out_row["four_dimension_errors"] = [message]
            failed.append((idx, message))
            rows.append(out_row)
            continue
        out_row["four_dimension_answer_field"] = answer_field
        sample = dict(out_row)
        sample["output"] = answer
        rows.append(out_row)
        tasks.append((idx, sample, args.input))

    script_counts: Dict[str, int] = {}
    for out_row in rows:
        script = str(out_row.get("four_dimension_script") or DEFAULT_SCRIPT)
        script_counts[script] = script_counts.get(script, 0) + 1

    system_prompt = build_system_prompt(args.script or DEFAULT_SCRIPT)
    print(
        f"[{os.path.basename(args.input)}] rows={len(rows)} "
        f"reused={len(done)} pending={len(tasks)} skipped={len(failed)} "
        f"scripts={script_counts} scale={SCORE_SCALE_LABEL} prompt=four_dimension"
    )

    started_at = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(score_row, task, args): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            idx = task[0]
            try:
                idx, result = future.result()
                rows[idx]["four_dimension"] = result
                status = "ok"
            except Exception as e:
                rows[idx].setdefault("four_dimension_errors", []).append(str(e))
                result = None
                status = "error"
            completed += 1
            save_json(args.output, {"data1": rows})
            scores_text = (
                " ".join(f"{name}={result['scores'][name]:g}" for name in DIMENSIONS)
                if result else "-"
            )
            print(
                f"[进度] {format_progress_bar(completed, len(tasks))} "
                f"idx={idx} status={status} "
                f"script={rows[idx].get('four_dimension_script', '')} {scores_text} "
                f"elapsed={format_elapsed(time.time() - started_at)}",
                flush=True,
            )

    summary = summarize(rows, args.input, args.judge_model)
    save_json(args.output, {
        "data1": rows,
        "four_dimension_summary": summary,
        "prompt": {
            "variant": "four_dimension",
            "scale": SCORE_SCALE_LABEL,
            "system_template": system_prompt,
        },
    })

    if args.output_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8", newline="\n") as f:
            for idx, row in enumerate(rows):
                f.write(json.dumps({
                    "row_index": idx,
                    "id": item_key(row, idx),
                    "instruction": row.get("instruction", ""),
                    "gold_char": row.get("gold_char", ""),
                    "answer_field": row.get("four_dimension_answer_field", ""),
                    "script": row.get("four_dimension_script", ""),
                    "reasoning": row.get(row.get("four_dimension_answer_field", ""), ""),
                    **({"scores": row["four_dimension"]["scores"],
                        "rationale": row["four_dimension"]["rationale"],
                        "judge_model": row["four_dimension"].get("judge_model", ""),
                        "judge_script": row["four_dimension"].get("script", "")}
                       if isinstance(row.get("four_dimension"), dict)
                       else {"scores": None, "error": row.get("four_dimension_errors", [])}),
                }, ensure_ascii=False) + "\n")

    summary_csv = os.path.splitext(args.output)[0] + "_summary.csv"
    write_summary_csv(summary_csv, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
