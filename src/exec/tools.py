"""
exec/tools.py —— 多工具抽象层（注册表 + 统一调用 + 审计）

参考 zavora-ai/computer-use-mcp 的「64 个工具统一注册 + 策略与审计层」，
以及 cc-haha 的「多工具 / 子智能体统一管理 + 权限模式」：

- **Tool 注册表**：每个能力是一个 Tool 对象（名字/描述/参数 schema/风险级别/handler），
  统一入口 `registry.call(name, **kwargs)`，不再散落成一堆方法。
- **可直接喂给 System2**：`registry.schema()` 输出工具清单，大模型做 function calling 时用，
  保证「大脑看到的能力」与「实际能执行的能力」**同源**（不同源就会调不存在的工具）。
- **风险级别**（对齐 cc-haha 的权限模式）：read / write / system，策略可按级别决定是否确认。
- **JSONL 审计**：每次调用落一条（时间/工具/参数/结果/耗时/后端），
  这既是排障依据，也是**经验层的数据来源**（预判 → 执行 → 实际结果 全链路可回溯）。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

# 项目根（src/exec/tools.py → src/exec → src → 根）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 审计默认落盘位置
DEFAULT_AUDIT = os.path.join("blobs", "exec_audit.jsonl")

# 参数里长文本截断长度（避免把整段输入/敏感内容写进日志）
_AUDIT_TEXT_MAX = 200


@dataclass
class ToolResult:
    ok: bool
    result: Any = None
    error: str = ""
    ms: float = 0.0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "result": self.result,
                "error": self.error, "ms": round(self.ms, 2)}


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    params: Dict[str, dict] = field(default_factory=dict)  # {名: {"type","desc","required"}}
    risk: str = "write"        # read / write / system
    enabled: bool = True

    def schema(self) -> dict:
        """OpenAI function-calling 风格的 schema（给 System2 用）。"""
        props, required = {}, []
        for k, v in (self.params or {}).items():
            props[k] = {"type": v.get("type", "string"),
                        "description": v.get("desc", "")}
            if v.get("required"):
                required.append(k)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": props,
                           "required": required},
        }


class ToolRegistry:
    """工具注册表：注册 → 校验 → 执行 → 审计。"""

    def __init__(self, audit_path: Optional[str] = DEFAULT_AUDIT,
                 backend_name: str = "", audit: bool = True,
                 auto_audit: bool = True):
        """audit     : 是否落盘审计日志（总开关）
           auto_audit : call() 时是否自动记一条。
                        执行层若已在自己的统一入口记审计，应设 False 避免重复。
        """
        self.tools: Dict[str, Tool] = {}
        self.audit_path = audit_path
        self.backend_name = backend_name
        self.audit_enabled = audit
        self.auto_audit = auto_audit
        if audit and audit_path:
            os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)

    # -- 注册 --
    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def register_fn(self, name: str, description: str,
                    params: Optional[Dict[str, dict]] = None,
                    risk: str = "write"):
        def deco(fn):
            self.register(Tool(name=name, description=description,
                               handler=fn, params=params or {}, risk=risk))
            return fn
        return deco

    # -- 清单（给大模型）--
    def schema(self, only_enabled: bool = True) -> List[dict]:
        return [t.schema() for t in self.tools.values()
                if (t.enabled or not only_enabled)]

    def names(self) -> List[str]:
        return [n for n, t in self.tools.items() if t.enabled]

    def describe(self) -> str:
        """人类可读清单（日志/调试用）。"""
        return "\n".join(
            f"  {t.name:16s} [{t.risk:6s}] {t.description}"
            for t in self.tools.values() if t.enabled)

    # -- 调用 --
    def call(self, name: str, **kwargs) -> ToolResult:
        tool = self.tools.get(name)
        t0 = time.perf_counter()
        if tool is None:
            r = ToolResult(False, error=f"UNKNOWN_TOOL:{name}")
        elif not tool.enabled:
            r = ToolResult(False, error=f"TOOL_DISABLED:{name}")
        else:
            missing = [k for k, v in (tool.params or {}).items()
                       if v.get("required") and k not in kwargs]
            if missing:
                r = ToolResult(False, error=f"MISSING_PARAMS:{missing}")
            else:
                try:
                    r = ToolResult(True, result=tool.handler(**kwargs))
                except Exception as e:
                    r = ToolResult(False, error=f"{type(e).__name__}: {e}")
        r.ms = (time.perf_counter() - t0) * 1000
        if self.auto_audit:
            self._audit(name, kwargs, r)
        return r

    # -- 审计（经验层数据源）--
    def _write(self, rec: dict) -> None:
        if not (self.audit_enabled and self.audit_path):
            return
        try:
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _safe_args(self, kwargs: dict) -> dict:
        safe = {}
        for k, v in (kwargs or {}).items():
            if isinstance(v, str) and len(v) > _AUDIT_TEXT_MAX:
                safe[k] = v[:_AUDIT_TEXT_MAX] + f"...<+{len(v) - _AUDIT_TEXT_MAX}chars>"
            else:
                safe[k] = v
        return safe

    def _audit(self, name: str, kwargs: dict, result: ToolResult) -> None:
        self._write({
            "ts": time.time(),
            "tool": name,
            "risk": (self.tools.get(name).risk if self.tools.get(name) else "?"),
            "args": self._safe_args(kwargs),
            "ok": result.ok,
            "error": result.error,
            "ms": round(result.ms, 2),
            "backend": self.backend_name,
        })

    def log_manual(self, action: str, args: Optional[dict] = None, ok: bool = True,
                   error: str = "", ms: float = 0.0, risk: str = "") -> None:
        """手动记一条审计。

        给「不经 registry.call、直接调执行方法」的场景用——保证无论走哪条路径，
        动作都被记录（审计是经验层的数据源，漏记等于经验有洞）。
        """
        self._write({
            "ts": time.time(),
            "tool": action,
            "risk": risk or (self.tools.get(action).risk
                             if self.tools.get(action) else "?"),
            "args": self._safe_args(args or {}),
            "ok": ok,
            "error": error,
            "ms": round(ms, 2),
            "backend": self.backend_name,
        })

    def recent(self, n: int = 20) -> List[dict]:
        """读最近 N 条审计（供经验层/排障用）。"""
        if not (self.audit_path and os.path.exists(self.audit_path)):
            return []
        try:
            with open(self.audit_path, "r", encoding="utf-8") as f:
                return [json.loads(l) for l in f.readlines()[-n:] if l.strip()]
        except Exception:
            return []
