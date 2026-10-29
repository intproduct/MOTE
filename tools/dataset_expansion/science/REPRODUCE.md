# 科学扩展流程

所有数据、缓存和临时目录位于 `D:\files\pydocs\MOTE\Datasets\expansion_20261016\science`。不运行来源仓库中的代码。

原始文件已保存且版本/SHA固定，通常无需重新下载。入口下载脚本为 `download.py`、`download_foundations.py`；其他官方网页的取得见对应 `.download.json` 和 `recover_openstax.py`。

使用已有 miniforge Python（requests、pyarrow）运行 `process.py textbook`、`process.py physicseval`、`process.py osr2`、`process.py nemotron`、`process_foundations.py`。不要同时运行多个机械盘重扫。

OpenStax 先运行 `select_openstax.py`，再用已有带 lxml 的 Codex bundled Python 运行 `recover_openstax.py`，最后以 miniforge Python 运行 `process_openstax.py`。恢复脚本使用官方 HTML 缓存；MathML 包括指数、分式及前置核素上下标。仅选择 PhysCorp-A 的 OpenStax 行，其他来源不导出。

`audit/manual_findings.json`、`audit/osr2_semantic_findings.json` 和 `audit/foundation_semantic_audit.json` 是内容哈希绑定的审查证据，不是应当随意重生成的训练标签。修改数据内容后，原审查结论必须重新核对。`audit_osr_semantics.py`、`audit_foundations.py` 保存本轮审查理由；不能把代码重复运行视为新的独立语义检查。

依次运行 `finalize.py`、`post_release.py`、`write_report.py`。只使用 `release/`；`normalized/` 是中间文件，不满足最终语义门禁。`post_release.py` 会将整个 OSR2 保持待审，并更新最终 SHA 和数量。

验证脚本是 `test_science.py`（miniforge Python，11 项）和 `test_openstax.py`（bundled Python，6 项）。最终全局去重、跨源题目分组、基准污染检查以及数据 split 由中央合并完成。

`verification_queue.py` 只制作 2,000 条物理盲审输入，不调用模型。源答案文件必须与盲审 teacher 输入隔离。不存在自动获得 2,000 条已验证新数据的步骤。
