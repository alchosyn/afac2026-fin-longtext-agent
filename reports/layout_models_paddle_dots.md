# PaddleOCR PP-StructureV3 与 dots.ocr 的版面分析质量（2026-09-30）

任务：对 10 篇文档跑 PaddleOCR 和 dots.ocr，判断版面分析质量——表格的错检漏检、表格分列是否正确、标题的错检漏检、公式的错检漏检。MinerU 4.0.4 和 Docling 2.129 的结果（见 `reports/parser_comparison_findings.md`）作为同页对照。

## 范围与设置

- **文档**：五类各取两篇——监管法规 csrc_0001_att1（16 页）、csrc_0002_att1（13）；研报 pack2_text01（26）、pack2_text03（36）；保险条款 ins_01（21）、ins_03（17）；金融合同 fc_text01（311）、fc_text02（253）；年报 fr_byd_2024（290）、fr_catl_2024（229）。
- **页面**：六篇短文档全页；四篇长文档按 52–58 页/篇分层抽样（含全部数学字体页、扫描页 5 页、有线框表的页 15 页、无线框表格页 8 页、有标题页 12 页、随机 10 页、前 2 页），共 **349 页**。抽样清单 `processed_data/_compare/_pages/sample_pages.json`。页面用 pymupdf 按 200 dpi 渲染成 PNG，两个模型吃同一张图。
- **PaddleOCR**：paddleocr 3.7.0 + paddlepaddle-gpu 3.3.1（cu126），`PPStructureV3` 默认模型（PP-DocLayout_plus-L 版面、PP-OCRv5 识别、SLANeXt 表格、PP-FormulaNet_plus-L 公式），关闭文档方向/去扭曲/文本行方向分类，其余默认；GPU 上 **14.8 秒/页**（法规页 4 秒，年报页 19 秒）。输出 `processed_data/<doc>/paddle.json`，脚本 `experiments/parser_comparison/run_paddle_structure.py`。
- **dots.ocr**：rednote-hilab/dots.ocr（HF 仓库现为 dots-studio/dots.ocr）bf16 权重 6.1 GB，transformers 4.56.1 后端。本机 RTX 4050 只有 6 GB 显存，bitsandbytes int8 量化能装下但解码只有 3.6 token/s，最终改成 **bf16 不量化、视觉编码器（2.4 GB）与语言层（2.7 GB）分阶段轮流上 GPU**，多页合批解码。仓库代码在 Windows 上有三处要绕：DotsVLProcessor 在 transformers 4.56 下缺 video_processor（手工构造 Qwen2_5_VLProcessor）、视觉注意力对单张图也构造 seq×seq 掩码（改为无掩码 cuDNN SDPA）、forward 在只给 inputs_embeds 时断言 input_ids。运行方式：**版面检测提示词（prompt_layout_only_en，只出框和类别）跑全部 349 页**，约 53 秒/页；**完整提示词（prompt_layout_all_en，带文字/HTML/LaTeX）只跑 49 个表格页**（`processed_data/_compare/_pages/table_pages.json`），用于分列检查，约 140 秒/页。输出 `dots.json` / `dots_all.json`，脚本 `experiments/parser_comparison/run_dots_ocr.py`。
- **速度提示**：这台机器当时是“节能”电源方案，GPU 满载只有 22 W（上限 60 W），bf16 矩阵乘实测 4.9 TFLOPS；切到高性能方案两个模型都会明显更快。
- **评估**：没有人工标注，`experiments/parser_comparison/compare_layout_models.py` 用四类参照并在四个模型间做共识：（1）pymupdf `find_tables(strategy="lines")` 找到的线框表及其行列数；（2）文档目录（PDF 书签，否则印刷目录，`docs.json`；研报目录里的“图1:”“表2:”条目是图表清单，已剔除）；（3）文本层里数学字体或数学符号所在的行；（4）块内的文本层文字（独有的“表格”像不像表、独有的“标题”有没有字体强调、独有的“公式”是不是普通文字）。三个以上模型都标出的元素某模型没标记为**共识漏检**，只有一个模型标出且没有参照支持的记为**独有块**（错检候选）。所有候选都截图人工复核（`processed_data/_compare/_review/`，判定记录 `manual_judgements.json`）。三模型汇总 `reports/layout_models_mdp.md/.json`，四模型汇总 `reports/layout_models_comparison.md/.json`。

## 一、表格

### 1. 错检与漏检（349 页，M = MinerU，D = Docling，P = PaddleOCR）

| | M | D | P |
|---|---|---|---|
| 表格块数 | 273 | 250 | 266 |
| 线框表召回（176 个线框表） | 98.9%（漏 2） | 96.6%（漏 6） | 98.9%（漏 2） |
| 共识表漏检 | 0 | 4 | 0 |
| 独有表格块（人工复核后确认为错检） | 7（4） | 1（0） | 3（2） |
| 拆分 / 合并 | 2 / 0 | 0 / 2 | 2 / 0 |

人工复核结论：

- **PaddleOCR 错检 2 处**，都在保险条款 ins_01（p3、p19）：条款号 + 标题 + 正文的两栏排版被当成表格。同类排版 MinerU 错检 3 处（p10、p15、p20），Docling 没有错检。
- **PaddleOCR 漏检 3 处**：fr_byd_2024 p226 下半页 2023 年的其他综合收益表——行标签列输出为 text，数字区域没有任何块覆盖（整块区域丢失）；pack2_text01 p14 页底两行的小表“表5”只找到了标题；fr_catl_2024 p36 跨页续表只剩一行时输出为 text。前两处 Docling 同样漏检。
- 同页 2024 年那张表 PaddleOCR 把行标签列拆成 text、数字部分才是 table（“拆分”2 处之一）。
- PaddleOCR 做对而别人错的：fr_byd_2024 p213 的无框参数表（MinerU 输出为公式）、pack2_text03 p19 蓝底样式表“表3”（MinerU 当图、Docling 当图）、p11 的“表1”（Docling 当图）。
- 线框表漏检 2 处里有 1 处是 pymupdf 把柱状图当成了线框（pack2_text03 p9，三个模型都正确地标为 chart/picture），所以真实漏检 P 只有 fr_catl_2024 p36 那一行。
- MinerU 错检 4 处：签章区域（fc_text01 p278）和上述保险条款两栏排版 3 处。Docling 漏检 6 处（借/贷分录框、无框年报表、两张样式表、续表行、表5）。

### 2. 分列

线框表的 pymupdf 列数不可靠（年报里三个模型的列数与它一致的只有 5%–22%，它会把合并单元格和空白间隔数成列），所以分列以人工复核为准，自动指标只看模型间一致性：

| 模型对 | 列数相同 | 结构 TEDS-S（抽样） |
|---|---|---|
| MinerU–Docling | 90.0%（250 对） | 0.921（181） |
| MinerU–PaddleOCR | 85.1%（262 对） | 0.927（186） |
| Docling–PaddleOCR | 87.4%（247 对） | 0.916（181） |

277 个表格簇里 215 个（78%）三方列数一致。从三方或线框不一致的 118 个簇里随机抽 37 个人工核对列数和单元格顺序：

| | 对 | 错 | 错法 |
|---|---|---|---|
| PaddleOCR | 27 | 10 | 多出空列 4（fc_text01 p293、fr_byd_2024 p75/p112、fr_catl_2024 p33）；少列 4（fr_byd_2024 p93 丢末列、fr_catl_2024 p126 16→14、pack2_text01 p17 排名与公司合并、p26 丢左栏）；**单元格顺序反转** 2（fr_catl_2024 p63 表头整行倒序、p222 行与格都倒序）；另有 2 张列数对但表头格错位（fr_byd_2024 p194、p226） |
| MinerU | 26 | 9（1 张无法判定） | 列合并 7（合同 fc_text01 p10/p46/p54 三张定义表和董监高表 4→2、5→4；ins_01 两栏排版；研报 pack2_text01 p14 6→4、p26 4→2）；多出空列 2（pack2_text03 p11 14→16、p27 4→5）；fc_text02 p145 列数对但单元格文字串行 |
| Docling | 25 | 4（8 张没参与） | 一律多一列（fc_text02 p129/p131/p142、fr_catl_2024 p189） |

PaddleOCR 单元格顺序反转不是偶发：262 张与 MinerU 配对的表里，130 张单元格序列完全一致，**4 张整个序列倒序**（fc_text01 p145、fr_byd_2024 p10/p219、fr_catl_2024 p222），另有表头行内倒序的（p63）没被这条自动规则抓到。倒序的表列数看着正确，但内容对不上列名，比多一列少一列更危险。

小结：三个模型分列正确率在这个“有分歧”样本上都在七成上下（PaddleOCR 73%、MinerU 72%、Docling 86% 但缺席 8 张），加上 78% 无分歧的表，整体大约九成以上。失败模式不同：MinerU 倾向把窄列合并（合同的“指”列、序号列），Docling 倾向多切一列，PaddleOCR 倾向多出空列、丢边缘列，并有单元格倒序。

## 二、标题

| | M | D | P |
|---|---|---|---|
| 标题块数 | 708 | 867 | 715 |
| 目录/书签召回（270 条，7 篇有目录） | 98.1% | 94.1% | 98.5% |
| 共识标题漏检 | 3 | 4 | 0 |
| 独有标题（其中文本层无字体强调） | 6（3） | 143（78） | 14（12） |
| 标题似句 | 4 | 13 | 6 |

人工复核：

- PaddleOCR 的 14 个独有标题里 11 个是真标题——主要在保险条款 ins_01（9.5/9.7/9.9/9.10 等条款标题，MinerU 把那片区域当成表格）和合同的小节标题（“3、前期差错更正及影响”“c、收取保证金”）；**错检 3 个**：年报的“□是 ☑否”勾选行、单独一个条款号“1.1.8”、研报里一句加粗的段首句。
- PaddleOCR 目录召回漏掉 4 条：fc_text01 p52“第一节 发行人基本情况…”、fr_byd_2024 p102 一条 47 字的长编号标题、p287“5、投资收益”、ins_03 p5“第十二条…”；MinerU 漏 5 条（含三处共识漏检：“（二）2023 年度财务报告…”“3、中介机构的情况”“6、担保情况…”），Docling 漏 16 条。
- MinerU 独有标题 6 个里 2 个错检（“图表：截至 2024 年末关联方应收款项”这个表格标题、一条“（2）语言能力完全丧失…”列表项）。Docling 独有 143 个，绝大多数是加粗段首句和图表标题被标成 section_header，与之前 15 篇的结论一致。

小结：PaddleOCR 的标题检出与 MinerU 同一水平（召回 98.5% 对 98.1%），漏检最少，错检 3 个；两者都比 Docling 稳。PaddleOCR 输出的是 paragraph_title/doc_title 两级，没有层级号，和 MinerU 一样需要后处理定层级。

## 三、公式

这十篇文档几乎没有独立公式：文本层里带数学字体或符号的 130 行，除 pack2_text03 p12 的两个居民消费函数外，都是正文里的 ≤、≥、×、±、☑、① 或者“资产负债率＝负债总额/资产总额×100%”这类文字公式，没有模型把它们当公式，也不应当。

| | M | D | P |
|---|---|---|---|
| 公式块数 | 3 | 2 | 2 |
| pack2_text03 p12 两个真公式 | 2 找到（第二块把上一行正文一起圈进来） | 2 | 2（LaTeX 正确，中文下标保留） |
| 错检 | 1（fr_byd_2024 p213 无框参数表输出成 LaTeX array） | 0 | 0 |
| 漏检 | 0 | 0 | 0 |

PaddleOCR 的 PP-FormulaNet 没有把表格、勾选框或带符号的正文误判成公式；在 p12 那页版面模型还给出了 6 个 formula 框（含行内公式），合并到段落后剩两个独立公式块。

## 四、dots.ocr

（批跑进行中，完成后补入：版面检测的表格/标题/公式错检漏检、49 个表格页的分列复核、速度。）

## 五、结论

（待 dots.ocr 结果补齐后写。）

## 附：文件

- 运行：`experiments/parser_comparison/run_paddle_structure.py`、`experiments/parser_comparison/run_dots_ocr.py`、`parsing/pagerender.py`；环境见记忆笔记 layout-models-paddle-dots-environment。
- 评估：`experiments/parser_comparison/compare_layout_models.py`（指标与复核截图）、`experiments/parser_comparison/review_sheets.py`（截图拼版）。
- 结果：`processed_data/<doc>/paddle.json`、`dots.json`、`dots_all.json`；`reports/layout_models_mdp.md/.json`（三模型）、`layout_models_comparison.md/.json`（四模型）；`processed_data/_compare/_review/`（截图、`manual_judgements.json`、`table_cols_sample.json`）。
