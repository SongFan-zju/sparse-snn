# 路径与校验和约定

新交接索引和导出元数据使用逻辑根。`${REPO_ROOT}`表示新checkout根目录，`${PYTHON}`表示使用者安装项目依赖的Python解释器，`${WEIGHTS_ROOT}`表示可选的外部权重目录；此包不含权重。`${CODE_BUNDLE_ROOT}`仅标识此次独立兼容检查的导出代码目录，合并到分支后就是`${REPO_ROOT}`。`${ORIGINAL_REPO_ROOT}`只表示第一次核验发生在原工作仓库，不是运行新分支所需的路径。

历史报告的PDF、TXT及 `report_source.json`原样保留，以维持原报告和SHA证据。它们以及原始推导笔记中的旧绝对路径属于历史环境记录，不能在新机器照抄执行。新环境执行命令以 `MATH_RESEARCH_INDEX_ZH.md`中的相对路径和 `python`为准。额外提供 `report_source.portable.json`，仅把旧路径替换成上述逻辑根，数学、数值与段落内容不变；它不是新报告或原报告的替换品。

`artifacts/papers/index.json`导出时将缓存PDF/TXT路径变为仓库相对路径，其 `_export_metadata.original_source_sha256`标识原始索引JSON。各论文条目内部的 `sha256`继续标识原缓存论文PDF，论文全文未随包提供。另一个原始 `index_phase_triangle.json`本已使用相对路径，保留原样。

`MATH_BUNDLE_MANIFEST.json`中每个条目的 `original_source_sha256`标识原始来源文件，`export_sha256`标识实际导出文件。原样复制时二者相同；路径规范化导出时二者可以不同，并记录 `transformation`；新增交接文档没有原始来源SHA。Manifest不对自身递归校验，外层最终仓库/提交可记录其SHA。

`MATH_SOURCE_REQUIREMENTS.json`采用不同的两列来源：`local_source_sha256`是筛选时本机源码指纹，`exported_code_sha256`是代码包实际模块指纹。二者差异已检查并记录，见 `MATH_CODE_COMPATIBILITY_ZH.md`，不能把前者当成导出匹配声明。
