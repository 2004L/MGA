"""
P0 验收工具：外部大模型连通性自检
================================
填完 .env.local 里的 MGA_LLM_API_KEY 后，跑这一条就知道大脑换成功了没：

    PYTHONPATH=src python -m check_llm

会打印：后端 / 模型名 / base_url / 密钥是否已配置 / 实际调用返回。
密钥只在本机读取，绝不打印。
"""

from __future__ import annotations

from perception.llm_bridge import build_llm, _cfg_runtime


def main() -> int:
    cfg = _cfg_runtime()
    backend = cfg.get("llm_backend", "mock")
    llm = build_llm(backend)

    model = getattr(llm, "model", "-")
    base_url = getattr(llm, "base_url", "-") or "-"
    has_key = bool(getattr(llm, "api_key", ""))

    print("== 大脑配置 ==")
    print(f"  后端     : {backend}")
    print(f"  模型     : {model}")
    print(f"  接口地址 : {base_url}")
    print(f"  密钥     : {'已配置' if has_key else '未配置'}")

    if backend in ("mock", "oracle"):
        print("\n  当前是本地后端，不调用外部 API。"
              "想用外部大模型请把 config.json 的 runtime.llm_backend 设为 openai。")
        return 0

    if not has_key:
        print("\n  请在项目根 .env.local 里填：MGA_LLM_API_KEY=你的Key")
        print("  （该文件已在 .gitignore，不会入库）")
        return 1

    print("\n== 连通性测试 ==")
    try:
        reply = llm.respond("只回复两个字：正常")
        text = (reply or "").strip()
        print(f"  返回: {text[:200]}")
        print("  结论: 连通 OK，大脑已换成外部大模型")
        return 0
    except Exception as e:
        print(f"  调用失败: {type(e).__name__}: {str(e)[:300]}")
        print("  排查: ① key 是否填对 ② base_url 是否正确 ③ 该账号是否开通了此模型")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
