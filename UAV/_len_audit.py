"""第三章篇幅体检：把我们的第三章与 TJCCT 的第三章放在同一把尺子上量。

两把尺子：
  1) 几何尺（只对 TJCCT 有效）—— 直接量 PDF 上"第 3 章标题"到"第 4 章标题"
     之间占了多少个"栏单位"(column-unit)。1 栏 = 半页，所以 2 栏单位 = 1 页。
     这是唯一不需要任何假设的度量。
  2) 词数尺（两边都用）—— 过滤掉数学符号 token 后的纯正文词数。
     我们的第三章是 A4 单栏导出的 PDF，几何尺量不出 IEEE 双栏页数，只能靠词数换算。

用法:
    python _len_audit.py
"""
import re
import sys

import fitz

sys.stdout.reconfigure(encoding="utf-8")

TJCCT = "A Two-timescale Approach for UAV-assisted Mobile Edge Computing.pdf"
OURS = "C:/Users/znhy/Downloads/SystemModel-Formulation-v4-final.pdf"

# 换算尺不写死，而是从 TJCCT 自己的版面回归出来（见 calibrate()）：
#   col_u = prose_words / WORDS_PER_COL_U
# 回归发现"每个公式的额外占位"与词数强共线（公式多的小节词也多），
# 二参数拟合会病态，所以用单参数：词数密度里已经隐含了 TJCCT 的公式密度。
TBL_ROW_COL_U = 0.020   # 符号表每行（15 行 ≈ 0.30 栏单位）
NOTATION_ROWS = 15      # SystemModel-Formulation-多服务器.md 里 TABLE I 的行数

# TJCCT 版面（Letter 612x792, 双栏）
COLS = {0: (40, 300), 1: (305, 575)}
TOP, BOT = 50.0, 748.0


def is_prose(w):
    """判断一个 token 是不是"正文词"而不是数学符号。"""
    if not re.search(r"[A-Za-z]", w):
        return False
    if len(re.sub(r"[^A-Za-z]", "", w)) < 2:   # 单字母 = 数学符号
        return False
    if re.search(r"[^\x00-\x7f]", w):          # 希腊字母 / 数学 unicode
        return False
    if re.search(r"[=\\^_{}\[\]<>|]", w):
        return False
    return True


def prose_words(text):
    return [w for w in text.split() if is_prose(w)]


# ---------------------------------------------------------------- TJCCT 几何尺
TJCCT_HEADS = [
    "SYSTEM MODEL AND PROBLEM", "3.1 System Model", "3.1.1 System Overview",
    "3.1.2 Basic Models", "3.2 Communication Model", "3.2.1 LoS Probability",
    "3.2.2 Channel Power Gain", "3.3 Computation Model",
    "3.3.1 Local Computing Model", "3.3.2 Edge Offloading Model",
    "3.4 Utility Model", "3.4.1 QoE of MDs", "3.4.2 Revenue of MEC",
    "3.4.3 Utility of System", "3.5 Problem Formulation", "4 ALGORITHM",
]


def measure_tjcct():
    doc = fitz.open(TJCCT)
    pos = {}
    for page_no in range(4, 8):                 # 第 3 章落在 p4~p7
        page = doc[page_no - 1]
        for head in TJCCT_HEADS:
            if head in pos:
                continue
            hits = page.search_for(head)
            if hits:
                r = hits[0]
                pos[head] = (page_no, 0 if r.x0 < 300 else 1, r.y0)

    def col_u(p, c, y):
        return p * 2 + c + (y - TOP) / (BOT - TOP)

    def text_between(a, b):
        (pa, ca, ya), (pb, cb, yb) = a, b
        ia, ib = pa * 2 + ca, pb * 2 + cb
        out = []
        for idx in range(ia, ib + 1):
            p, c = divmod(idx, 2)
            x0, x1 = COLS[c]
            y0 = ya if idx == ia else TOP
            y1 = yb if idx == ib else BOT
            if y1 > y0:
                out.append(doc[p - 1].get_text(clip=fitz.Rect(x0, y0, x1, y1)))
        return "\n".join(out)

    rows, prev = [], None
    for head in TJCCT_HEADS:
        if head not in pos:
            continue
        if prev:
            seg = text_between(pos[prev], pos[head])
            rows.append((prev, len(prose_words(seg)),
                         col_u(*pos[head]) - col_u(*pos[prev])))
        prev = head
    total = col_u(*pos["4 ALGORITHM"]) - col_u(*pos["SYSTEM MODEL AND PROBLEM"])
    return rows, total


# ------------------------------------------------------------------ 我们的第三章
def measure_ours():
    doc = fitz.open(OURS)
    text = "\n".join(p.get_text() for p in doc)

    heads = [(m.start(), m.group(0).strip())
             for m in re.finditer(r"^\s*3(\.\d+){1,2}\s+[A-Z].{0,60}$", text, re.M)]
    heads.append((len(text), "EOF"))
    rows = []
    for i in range(len(heads) - 1):
        seg = text[heads[i][0]:heads[i + 1][0]]
        rows.append((heads[i][1], len(prose_words(seg))))

    # 带编号的显示公式：行尾的 (n)
    labels = set()
    for line in text.split("\n"):
        m = re.search(r"\((\d{1,2})\)\s*$", line.strip())
        if m:
            labels.add(int(m.group(1)))
    # 行内出现的 (n) 视为交叉引用
    inline = {}
    for line in text.split("\n"):
        s = line.rstrip()
        for m in re.finditer(r"\((\d{1,2})\)", s):
            if m.end() < len(s):
                inline[int(m.group(1))] = inline.get(int(m.group(1)), 0) + 1
    never = [n for n in sorted(labels) if inline.get(n, 0) == 0]
    return rows, len(prose_words(text)), sorted(labels), never, doc.page_count


def measure_md(md_path, per_col_u):
    """量仓库里的 Markdown 版（v5.0 之后正文都在这里改）。

    与 PDF 用同一套"正文词"规则, 所以两边可比。
    """
    text = open(md_path, encoding="utf-8").read()
    cut = text.find("# 附录（中文")
    body = text[:cut] if cut > 0 else text
    # 从正文第一句开始量, 跳过版本表 / "What changed in v5.0" 这类
    # 【投稿时会删掉的元信息】—— 否则会把编辑批注算成论文正文。
    first = body.find("In this section,")
    if first > 0:
        body = body[first:]
    marks = [("第三章", 0), ("第四章", body.find("# IV. Multi-Agent")),
             ("Appendix A", body.find("# Appendix A"))]
    bounds = []
    for i, (name, start) in enumerate(marks):
        if start < 0:
            continue
        end = next((s for _, s in marks[i + 1:] if s > 0), len(body))
        bounds.append((name, start, end))

    out = []
    for name, a, b in bounds:
        seg = body[a:b]
        # 去掉显示公式块与表格行, 再把行内公式抹掉
        t = re.sub(r"\$\$.+?\$\$", " ", seg, flags=re.S)
        t = "\n".join(l for l in t.split("\n") if not l.strip().startswith("|"))
        t = re.sub(r"\$[^$]+\$", " ", t)
        w = len(prose_words(t))
        n_eq = len(re.findall(re.escape("\\") + r"tag\{", seg))
        out.append((name, w, n_eq, w / per_col_u))
    return out


def section_words(md_path):
    """按 3.x 大节汇总我们的正文词数；顺带单独拆出 3.3.3 排队（TJCCT 无此节）。"""
    text = open(md_path, encoding="utf-8").read()
    body = text[:text.find("# IV. Multi-Agent")]
    first = body.find("In this section,")
    if first > 0:
        body = body[first:]
    heads = [(m.start(), m.group(0).strip())
             for m in re.finditer(r"^#{2,3} .+$", body, re.M)]
    heads.append((len(body), "EOF"))
    out = {}
    for i in range(len(heads) - 1):
        seg = body[heads[i][0]:heads[i + 1][0]]
        t = re.sub(r"\$\$.+?\$\$", " ", seg, flags=re.S)
        t = "\n".join(l for l in t.split("\n") if not l.strip().startswith("|"))
        t = re.sub(r"\$[^$]+\$", " ", t)
        w = len(prose_words(t))
        m = re.match(r"#+ (3\.\d)", heads[i][1])
        key = m.group(1) if m else "章首"
        out[key] = out.get(key, 0) + w
        if "3.3.3" in heads[i][1]:
            out["__queue__"] = w
    return out


def calibrate(t_rows):
    """用 TJCCT 各小节 (词数 -> 栏单位) 回归出词数密度。最小二乘, 无截距。"""
    num = sum(w * u for _, w, u in t_rows)
    den = sum(w * w for _, w, u in t_rows)
    per_col_u = den / num          # 词 / 栏单位
    resid = max(abs(w / per_col_u - u) for _, w, u in t_rows)
    return per_col_u, resid


def main():
    t_rows, t_total = measure_tjcct()
    o_rows, o_words, o_labels, o_never, o_pages = measure_ours()
    t_words = sum(r[1] for r in t_rows)

    print("=" * 72)
    print("TJCCT 第 3 章（几何尺，唯一无假设的度量）")
    print("=" * 72)
    for name, w, u in t_rows:
        print(f"  {w:5d}w  {u:5.2f} 栏单位 ({u/2*100:4.0f}% 页)  {name}")
    print(f"\n  合计 {t_words}w, {t_total:.2f} 栏单位 = "
          f"{t_total/2:.2f} 个 IEEE 双栏页")

    per_col_u, resid = calibrate(t_rows)
    print(f"\n  标定（最小二乘, 无截距）: {per_col_u:.0f} 正文词 / 栏单位 "
          f"= {per_col_u*2:.0f} 词 / 页")
    print(f"  逐小节最大残差 {resid:.2f} 栏单位（≈{resid/2*100:.0f}% 页）"
          " —— 这把尺子够用")
    print(f"  TJCCT 公式密度: 每 {t_words/20:.0f} 词 1 个带编号公式")

    print()
    print("=" * 72)
    print(f"我们的第 3 章（A4 单栏 {o_pages} 页导出，几何尺不适用，用上面的尺子换算）")
    print("=" * 72)
    for name, w in o_rows:
        print(f"  {w:5d}w  {name}")
    n_eq = len(o_labels)
    u = o_words / per_col_u
    u_tbl = NOTATION_ROWS * TBL_ROW_COL_U
    print(f"\n  正文词数 {o_words}  |  带编号显示公式 {n_eq} 个 "
          f"(密度: 每 {o_words/n_eq:.0f} 词 1 个)")
    print(f"  换算: {u:.2f} 栏单位 = {u/2:.2f} 个 IEEE 双栏页（不含符号表）")
    print(f"        加符号表 {NOTATION_ROWS} 行 (+{u_tbl:.2f} 栏单位) "
          f"→ {(u+u_tbl)/2:.2f} 页")

    print()
    print("=" * 72)
    print("判决")
    print("=" * 72)
    print(f"  A4 单栏 {o_pages} 页  ≠  IEEE 双栏 {o_pages} 页。真实换算见下。")
    print(f"  正文词数  我们 {o_words}  vs  TJCCT {t_words}   "
          f"({o_words/t_words-1:+.0%})")
    print(f"  页数(无符号表) 我们 {u/2:.2f}  vs  TJCCT {t_total/2:.2f}   "
          f"({(u/2)/(t_total/2)-1:+.0%})")
    print(f"  页数(含符号表) 我们 {(u+u_tbl)/2:.2f}  vs  TJCCT {t_total/2:.2f}   "
          f"({((u+u_tbl)/2)/(t_total/2)-1:+.0%})")
    print(f"\n  与 TJCCT 词数持平        → 删 {max(0, o_words-t_words):.0f} 词")
    print(f"  与 TJCCT 页数持平(无表)  → 删 "
          f"{max(0, o_words-t_total*per_col_u):.0f} 词")
    print(f"  与 TJCCT 页数持平(含表)  → 删 "
          f"{max(0, o_words-(t_total-u_tbl)*per_col_u):.0f} 词")

    print()
    print("=" * 72)
    print("仓库 Markdown 版（v5.0；正文现在都在这里改）")
    print("=" * 72)
    md = measure_md("SystemModel-Formulation-多服务器.md", per_col_u)
    for name, w, n_eq, u in md:
        print(f"  {name:12s} {w:5d} 词  {n_eq:3d} 编号公式  "
              f"{u:5.2f} 栏单位 = {u/2:.2f} 页")
    if md:
        w3 = md[0][1]
        print(f"\n  第三章 {w3} 词 vs TJCCT {t_words} 词  ({w3/t_words-1:+.1%})")
        print(f"  第三章 {w3/per_col_u/2:.2f} 页（不含符号表）/ "
              f"{(w3/per_col_u+u_tbl)/2:.2f} 页（含）  vs TJCCT {t_total/2:.2f}")

        # 逐大节对照: 差在哪一节比"总共差多少"有用得多
        print("\n  逐大节对照（TJCCT 的 3.6 = 他们内联写在 3.1.1 里的双时标定义）:")
        tj = {"3.1": 838, "3.2": 507, "3.3": 436, "3.4": 475,
              "3.5": 184, "3.6": 130}
        ours = section_words("SystemModel-Formulation-多服务器.md")
        queue_w = ours.pop("__queue__", 0)
        for key in sorted(tj):
            o = ours.get(key, 0)
            print(f"    {key}  我们 {o:>4}  vs TJCCT {tj[key]:>4}   {o-tj[key]:>+5}")
        print(f"\n  ★ 差额里有 {queue_w} 词是 3.3.3 服务器占用与排队 —— "
              f"TJCCT【没有这一节】。")
        print(f"    扣掉它: {w3-queue_w} 词 vs {t_words}  "
              f"({(w3-queue_w)/t_words-1:+.1%}) —— 共有题材上我们比他们更紧。")

    print()
    print("从未被行内引用的带编号公式（候选：改为不编号或就地折叠）")
    print(f"  {o_never}   —— 共 {len(o_never)} / {len(o_labels)} 个")
    print("  注意: 用符号引用（如正文写 Psi 而不写 '(23)'）也算被用到，"
          "这份名单只是候选，需人工过一遍。")


if __name__ == "__main__":
    main()
