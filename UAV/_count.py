"""量一下第三章的篇幅, 并与 TJCCT 的第 3 章对照。全用 raw 字符串, 避免 \t \r 被转义。"""
import io
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

W_PER_PAGE = 1050      # IEEE Trans 双栏 10pt, 满页纯文字的词数
EQ_PAGE = 0.024        # 每个带编号显示公式约占页面比例 (3~4 行 + 上下留白)
TBL_ROW = 0.010        # 每个表格行


def measure(body, label):
    eq_blocks = re.findall(r"\$\$(.+?)\$\$", body, re.S)
    tagged = re.findall(r"\\tag\{", body)
    tbl_rows = [l for l in body.split("\n") if l.strip().startswith("|")]

    txt = re.sub(r"\$\$.+?\$\$", " ", body, flags=re.S)
    txt = "\n".join(l for l in txt.split("\n")
                    if not l.strip().startswith("|"))
    txt = re.sub(r"\$[^$]+\$", " X ", txt)
    txt = re.sub(r"[#>*`|_-]", " ", txt)
    words = [w for w in txt.split() if re.search(r"[A-Za-z]", w)]

    est = (len(words) / W_PER_PAGE + len(tagged) * EQ_PAGE
           + len(tbl_rows) * TBL_ROW)
    print(f"=== {label} ===")
    print(f"  正文词数            : {len(words):5d}")
    print(f"  显示公式块          : {len(eq_blocks):5d}   带编号 {len(tagged)}")
    print(f"  表格行              : {len(tbl_rows):5d}")
    print(f"  IEEE 双栏 10pt 估算 : {est:.1f} 页")
    print(f"     文字 {len(words)/W_PER_PAGE:.2f} | "
          f"公式 {len(tagged)*EQ_PAGE:.2f} | 表格 {len(tbl_rows)*TBL_ROW:.2f}")
    print()
    return len(words), len(tagged), est


s = io.open("SystemModel-Formulation-多服务器.md", encoding="utf-8").read()
cut = s.find("# 附录（中文")
body = s[:cut] if cut > 0 else s

# 拆成 3.x / IV 两段, 看各占多少
iv = body.find("# IV. Multi-Agent Reinforcement Learning")
sec3 = body[:iv] if iv > 0 else body
sec4 = body[iv:] if iv > 0 else ""

measure(body, "全文 (第三章 + 第四章 MARL)")
measure(sec3, "只算第三章 (3.1 ~ 3.6)")
if sec4:
    measure(sec4, "只算第四章 (MARL 4.1 ~ 4.4)")

# 逐小节
print("=== 第三章各小节词数占比 ===")
heads = [(m.start(), m.group(0).strip())
         for m in re.finditer(r"^#{2,3} .+$", sec3, re.M)]
heads.append((len(sec3), "EOF"))
rows = []
for i in range(len(heads) - 1):
    seg = sec3[heads[i][0]:heads[i + 1][0]]
    t = re.sub(r"\$\$.+?\$\$", " ", seg, flags=re.S)
    t = "\n".join(l for l in t.split("\n")
                  if not l.strip().startswith("|"))
    t = re.sub(r"\$[^$]+\$", " X ", t)
    t = re.sub(r"[#>*`|_-]", " ", t)
    w = len([x for x in t.split() if re.search(r"[A-Za-z]", x)])
    ntag = len(re.findall(r"\\tag\{", seg))
    rows.append((heads[i][1][:52], w, ntag))
tot = sum(r[1] for r in rows) or 1
for name, w, ntag in sorted(rows, key=lambda r: -r[1]):
    bar = "#" * int(w / tot * 60)
    print(f"  {w:5d}w  {ntag:2d}eq  {w/tot*100:4.1f}%  {bar}  {name}")
