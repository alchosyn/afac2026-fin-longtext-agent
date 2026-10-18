# -*- coding: utf-8 -*-
"""Check agent.decide.direction (which options a stem asks for) without any model call.

  1. real      every choice stem of the B list and of group A: the direction, the clause and
               the word it was read from; a stem with a judgement word that is not read is listed
  2. twins     the judgement word of every real stem replaced by its negations (正确 -> 错误 /
               不正确 / 有误 ...) must ask for 错误, replaced by 没有错误 for 正确
  3. topics    negative words put into the topic of every real stem must change nothing
  4. cases     constructed stems with the direction they must give

Usage:
    python tools/check_direction.py          # the summary and what failed
    python tools/check_direction.py -v       # every stem and case
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.decide import WORD, direction     # noqa: E402
from agent.scope import load_b_questions     # noqa: E402

W, R = "错误", "正确"
NEGATED = {"正确": ["错误", "不正确", "有误"], "准确": ["不准确", "错误"], "符合": ["不符合"], "一致": ["不一致"],
           "成立": ["不成立"], "合理": ["不合理"], "恰当": ["不恰当"], "相符": ["不相符"]}
TOPICS = ["关于年龄错误、登记信息不一致或披露不准确的情形，", "某机构发现报告数据有误且与原文不符。", "对于不正确、不合理的定价，"]
INSIDE = ["关于年龄错误处理的", "涉及信息披露不一致的"]              # put before the noun of the question clause
CASES = [
    # the question is not the last clause
    ("以下哪些判断错误？计算时使用报告中的原始金额，最终结果四舍五入至两位小数", W),
    ("以下哪些判断正确？计算时使用报告中的原始金额，最终结果四舍五入至两位小数", R),
    ("以下说法错误的是？请从下列选项中选择。", W), ("以下说法错误的是？请选出所有符合题意的选项", W),
    ("以下说法错误的是？请选出正确答案", W), ("关于锁定期，下列说法错误的是，结果保留两位小数", W),
    ("以下哪些判断错误（计算时使用原始金额，结果保留两位小数）", W),
    ("以下哪些判断正确（2024年数据采用2024年年报原始披露值（即2025年年报的调整前口径））", R),
    # the question is cut by a comma
    ("下列说法中，错误的是", W), ("下列各项表述中，不正确的有哪些？", W), ("以下关于这些案例的分析中，哪些是错误的", W),
    ("以下关于这些案例的分析中，哪些是正确的", R), ("关于科源制药重组，错误的是：", W), ("下列各项中，说法有误的是", W),
    ("下列各项，哪一项有误", W), ("下列选项中，说法不正确的是", W),
    # forms of asking
    ("以下说法错误的是", W), ("下列说法中不正确的有哪些？", W), ("下列表述有误的是（ ）。", W), ("下列说法错误的一项是", W),
    ("以下错误的说法是", W), ("下列哪项是错误的", W), ("下列哪个选项的说法是错误的", W), ("以下哪些说法不正确", W),
    ("以下哪些说法存在错误", W), ("以下存在错误的是", W), ("下列说法“错误”的是", W), ("以下说法中**不正确**的是", W),
    ("请选出下列说法中错误的一项", W), ("关于该条款，请指出其中不正确的说法", W), ("请选出错误的一项", W),
    ("以下说法不完全正确的是", W), ("下列表述并不准确的是", W), ("以下哪些分析不合理", W), ("以下哪些结论不成立", W),
    ("以下哪些计算结果错误", W), ("下列产品匹配错误的是", W), ("则三者退保所得金额排序错误的是", W),
    ("则各自身故保险金金额排序不正确的是", W), ("下列对应关系错误的是", W), ("以下哪些公司的资产负债率计算错误", W),
    ("以下哪些信息是不准确的", W), ("以下哪些数据是错误的", W), ("以下哪些数据错误", W), ("以下哪些做法是错误的", W),
    ("下列做法不恰当的是", W), ("以下哪项描述错误反映了其中关于违约赔偿或发行规模的细节", W), ("以下哪些说法错误地理解了条款", W),
    ("哪些是不准确或缺乏依据的", W), ("上述说法中未必正确的是", W), ("以下说法并非准确的有", W), ("以下哪项不是正确的", W),
    ("以下哪些分析是不合理的", W), ("以下哪些判断最不合理", W), ("哪两项最不准确", W), ("下列说法哪些不对", W),
    ("下列说法不对的是", W), ("以下判断中错误的有几项", W), ("下面说法哪些是不正确的", W), ("下列选项错误的是（ ）", W),
    ("以下选项错误的是", W), ("错误的是", W), ("其中表述不正确的是", W), ("以下哪一说法是错误的", W), ("下列说法哪项错误", W),
    ("下列说法 错误 的是 ____", W), ("下列说法错误的是——", W), ("下列说法不正确的是哪些选项", W), ("【多选】下列说法错误的是", W),
    ("根据比亚迪 2025 年年度报告，以下哪些判断 不正确？", W),
    # 符合 / 一致 said of a statement
    ("关于锁定期安排，下列说法与原文不一致的是：", W), ("以下哪些说法与募集说明书的约定不一致", W), ("与原文不符的是", W),
    ("下列哪项不符合监管要求", W), ("下列哪项表述不符合监管要求", W), ("以下说法不符合三份募集说明书规定的是", W),
    ("以下哪些最不符合当前行业的主流观点", W), ("下列描述与文档内容不相符的有", W), ("以下哪些数据与年报不一致", W),
    ("下列与条款规定不一致的是", W), ("以下哪些与原文不一致", W),
    # asking for 正确
    ("下列说法没有错误的是", R), ("以下表述不存在错误的是", R), ("下列说法正确无误的是", R), ("正确的是", R),
    ("以下哪些陈述最准确地概括了它们各自的核心应对逻辑", R), ("哪些是准确且有充分依据的", R), ("该表述是否正确？", R),
    ("以下哪些信息是准确的", R), ("以下哪项描述准确反映了其中关于违约赔偿或发行规模的细节", R), ("以下哪些分析是合理的", R),
    ("下列说法是否存在错误", R),
    # the topic holds the negative word
    ("金融机构发现登记信息错误、不一致或不完整。下列说法正确的是", R), ("下列关于年龄错误处理的说法正确的是", R),
    ("关于集中度指标不符合监管要求的情况，以下说法正确的是", R), ("以下关于年龄错误处理的说法中，正确的是", R),
    ("以下关于年龄错误处理的说法中，错误的是", W), ("下列关于信息披露不准确的处罚的说法不正确的是", W),
    ("关于年龄错误的处理，说法正确的是", R), ("关于年龄错误的处理，说法错误的是", W), ("以下对年龄错误的处理正确的是", R),
    ("以下关于信息披露规定的表述中，不准确的是", W), ("下列涉及信息披露的说法错误的是", W),
    ("关于“年龄错误导致少付保费后，已发生保险事故时按实付/应付比例给付或赔付”的规定，下列产品匹配正确的是", R),
    ("以下哪些产品约定了年龄错误的处理方式", R), ("以下哪些产品对年龄错误的处理有明确约定", R), ("以下哪些情形属于年龄错误", R),
    ("下列产品中约定了年龄错误处理方式的是", R), ("下列情形中属于年龄错误的是", R), ("下列哪些情况会导致信息披露不准确", R),
    ("下列报告中存在计算错误的是", R), ("以下哪些报告存在计算错误", R), ("以下哪些产品明确写有“与事实不符”的免责情形", R),
    ("根据《关于纠正不正确、不一致披露的规定》，下列说法正确的是", R), ("哪些因素导致业绩预测不准确？以下分析合理的是", R),
    ("市场此前的判断错误在哪里？以下哪些分析最合理", R), ("下列属于数据错误的是", R), ("下列属于登记信息错误的是", R),
    ("以下哪些信息披露不准确", R), ("下列各项中，年报披露有误的是", R), ("以下哪些属于错误的操作", R),
    ("以下哪些判断被证明是错误的", R), ("下列哪些条款不对称地保护了投保人", R),
    # a negation that belongs to the statement
    ("下列不属于责任免除情形的是", R), ("以下哪些情形下，保险人不承担保险责任", R), ("以下哪些产品不符合投保条件", R),
    ("下列产品中不符合投保年龄要求的是", R), ("以下哪些情形与条款约定不一致", R), ("下列产品中与平安条款不一致的是", R),
    ("以下哪些产品的犹豫期约定不一致", R), ("以下哪些指标的计算口径不一致", R), ("下列哪项处理不符合规定", R),
    ("以下哪些产品的条款中明确将“自杀”列为责任免除，但规定了2年后的例外", R),
    # no judgement word
    ("用于债务结构调整的金额为多少亿元", R), ("对分类评价是否可能有影响", R), ("下列产品明确列明该项的是", R),
]


def real_stems():
    stems = [(q["qid"], q["question"].strip()) for q in load_b_questions() if q["type"] in ("单选题", "多选题")]
    for f in sorted((ROOT / "questions" / "group_a").glob("*.json")):
        stems += [(q["qid"], q["question"].strip()) for q in json.load(open(f, encoding="utf-8")) if q.get("answer_format") in ("mcq", "multi")]
    return stems


def main(argv):
    verbose = "-v" in argv
    stems, failed = real_stems(), 0

    tally = Counter()
    print(f"== 1. real: {len(stems)} choice stems")
    for qid, stem in stems:
        asked, clause, word = direction(stem)
        tally[(asked, "判断词" if word else "默认")] += 1
        unread = word is None and WORD.search(re.split(r"[，,。；;：:？?！!\n]", stem.rstrip("？?。 "))[-1])
        if verbose or asked == W or unread:
            print(f"  {qid:<11}{asked} {word}  “{clause[-40:]}”" + ("   <- 末句有判断词但没有读成问法" if unread else ""))
    print("  ", {f"{a}/{b}": n for (a, b), n in sorted(tally.items())})

    made = bad = 0
    for qid, stem in stems:
        word = direction(stem)[2]
        if word not in NEGATED:
            continue
        i = stem.rfind(word)
        for new, want in [(n, W) for n in NEGATED[word]] + [("没有错误", R)]:
            twin = stem[:i] + new + stem[i + len(word):]
            made += 1
            if direction(twin)[0] != want:
                bad += 1
                print(f"  WRONG {qid} {direction(twin)} <- …{twin[-50:]}")
    print(f"== 2. twins: {made}, wrong {bad}")
    failed += bad

    made = bad = 0
    for qid, stem in stems:
        base = direction(stem)[0]
        variants = [t + stem for t in TOPICS]
        for noun in ("说法", "判断", "描述", "产品", "情形", "陈述", "分析"):
            if noun in stem:
                j = stem.rfind(noun)
                variants += [stem[:j] + t + stem[j:] for t in INSIDE]
        for v in variants:
            made += 1
            if direction(v)[0] != base:
                bad += 1
                print(f"  CHANGED {qid} {direction(v)} <- …{v[-60:]}")
    print(f"== 3. topics: {made} variants, changed {bad}")
    failed += bad

    bad = 0
    for stem, want in CASES:
        asked, clause, word = direction(stem)
        bad += asked != want
        if asked != want or verbose:
            print(f"  {'ok   ' if asked == want else 'WRONG'} {asked} {word} “{clause}”  <- {stem}")
    print(f"== 4. cases: {len(CASES)}, wrong {bad}")
    failed += bad
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
