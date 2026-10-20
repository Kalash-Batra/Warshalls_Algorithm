"""
=====================================================================
 Implementation and Visualization of Warshall's Algorithm
 for Transitive Closure              (Discrete Mathematics - CIA)
=====================================================================

WHAT THIS PROGRAM DOES
  Given a SET A and a RELATION R on A (pairs like (a, b)), it finds the
  TRANSITIVE CLOSURE R+ using Warshall's algorithm, shows every step in
  the console, and saves:
      warshall_animation.gif     -> full step-by-step animation
      warshall_before_after.png  -> input graph vs transitive closure
      warshall_steps.png         -> matrix after every stage (one page)

HOW TO USE
  Run:  python warshall_interactive.py

  Menu:
     [Enter]  -> runs the DEMO from the PPT
                 Set A = {1, 2, 3, 4}   R = {(1,2), (2,3), (3,4), (4,1)}
     [c]      -> enter YOUR OWN set and relation

Install once:  pip install matplotlib networkx pillow
"""

import math
import re

import matplotlib
matplotlib.use("Agg")  # saves files; no pop-up window needed
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Arc, FancyArrowPatch

# ---------------- colours ----------------
C_NODE, C_NODE_K, C_NODE_EDGE = "#a8dadc", "#f4a261", "#1d3557"
C_EDGE, C_PATH, C_NEW = "#457b9d", "#f77f00", "#e63946"
C_ONE, C_NEWCELL, C_LINE = "#d8f3dc", "#ff9aa2", "#fff3b0"

# ---------------- the PPT demo ----------------
DEMO_SET = ["1", "2", "3", "4"]
DEMO_REL = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "1")]


# =====================================================================
# 1. ALGORITHM  (records every event so we can print AND animate)
# =====================================================================
def warshall(matrix):
    """Mk[i][j] = Mk-1[i][j] OR (Mk-1[i][k] AND Mk-1[k][j]).
    Returns (closure, frames, stages)."""
    n = len(matrix)
    M = [r[:] for r in matrix]
    frames, stages = [], [{"k": 0, "matrix": [r[:] for r in M], "new": []}]

    def snap(kind, k, new, focus=None, text="", sub=""):
        frames.append({"kind": kind, "k": k, "matrix": [r[:] for r in M],
                       "new": list(new), "focus": focus, "text": text, "sub": sub})

    snap("start", 0, [], text="M0 : initial matrix of the relation R",
         sub="Only the pairs given in R (direct edges).")

    for k in range(n):
        stage_new = []
        snap("intro", k + 1, stage_new, text=f"@K{k}@ is now allowed as intermediate",
             sub="Look for pairs where  i -> k  and  k -> j  already exist.")
        for i in range(n):
            for j in range(n):
                if not M[i][j] and M[i][k] and M[k][j]:
                    M[i][j] = 1                                  # Warshall update
                    stage_new.append((i, j))
                    snap("found", k + 1, stage_new, focus=(i, j, k),
                         text=f"NEW pair  @I{i}@ -> @I{j}@",
                         sub=f"Because @I{i}@ -> @I{k}@ and @I{k}@ -> @I{j}@ already exist.")
        kind = "final" if k == n - 1 else "stage_end"
        text = ("FINAL TRANSITIVE CLOSURE" if k == n - 1 else f"Stage {k+1} complete")
        sub = ("New this stage: " + ", ".join(f"@I{a}@->@I{b}@" for a, b in stage_new)
               if stage_new else "No new pairs at this stage.")
        snap(kind, k + 1, stage_new, text=text, sub=sub)
        stages.append({"k": k + 1, "matrix": [r[:] for r in M], "new": list(stage_new)})
    return M, frames, stages


def fill_labels(text, labels):
    """Turn @K3@ / @I2@ placeholders into real element names."""
    text = re.sub(r"@K(\d+)@", lambda m: labels[int(m.group(1))], text)
    text = re.sub(r"@I(\d+)@", lambda m: labels[int(m.group(1))], text)
    return text


# =====================================================================
# 2. CONSOLE OUTPUT
# =====================================================================
def fmt_set(labels):
    return "{" + ", ".join(labels) + "}"


def fmt_rel(M, labels):
    pairs = [f"({labels[i]},{labels[j]})" for i in range(len(M)) for j in range(len(M)) if M[i][j]]
    return "{" + ", ".join(pairs) + "}"


def print_matrix(M, labels, new=(), title=""):
    n = len(M)
    lw = max(len(s) for s in labels)
    cw = max(lw, 3)
    if title:
        print(title)
    print(" " * (3 + lw) + " ".join(s.center(cw) for s in labels))
    for i in range(n):
        cells = []
        for j in range(n):
            c = f"[{M[i][j]}]" if (i, j) in new else f" {M[i][j]} "
            cells.append(c.center(cw))
        print("  " + labels[i].ljust(lw) + " " + " ".join(cells))
    print()


def print_all(stages, closure, labels, original):
    print("=" * 64)
    print(" WARSHALL'S ALGORITHM - STEP BY STEP")
    print("=" * 64)
    print(f"\nSet A      = {fmt_set(labels)}")
    print(f"Relation R = {fmt_rel(original, labels)}\n")
    print_matrix(stages[0]["matrix"], labels, title="Initial matrix M0 (the relation R):")
    for st in stages[1:]:
        k = st["k"]
        print_matrix(st["matrix"], labels, set(st["new"]),
                     title=f"After k = {k}   ({labels[k-1]} allowed as intermediate)")
        if st["new"]:
            for i, j in st["new"]:
                print(f"    NEW: ({labels[i]},{labels[j]})   "
                      f"because ({labels[i]},{labels[k-1]}) and ({labels[k-1]},{labels[j]}) are in R")
        else:
            print("    No new pairs at this stage.")
        print("    ( [x] = newly added entry )\n")
    print("=" * 64)
    print_matrix(closure, labels, title="FINAL reachability matrix (transitive closure):")
    print(f"Transitive closure  R+ = {fmt_rel(closure, labels)}")
    added = [(i, j) for i in range(len(closure)) for j in range(len(closure))
             if closure[i][j] and not original[i][j]]
    print(f"Pairs added by Warshall = "
          + ("{" + ", ".join(f"({labels[i]},{labels[j]})" for i, j in added) + "}" if added
             else "none (R was already transitive)"))
    print("=" * 64 + "\n")


# =====================================================================
# 3. DRAWING
# =====================================================================
def positions(n):
    return nx.circular_layout(nx.empty_graph(n), scale=1.0)


def draw_edge(ax, pos, u, v, color, width, rad=0.18, alpha=1.0, z=2):
    if u == v:
        x, y = pos[u]
        nrm = math.hypot(x, y) or 1.0
        cx, cy = x + x / nrm * 0.17, y + y / nrm * 0.17
        ax.add_patch(Arc((cx, cy), 0.26, 0.26, theta1=0, theta2=330,
                         color=color, lw=width, alpha=alpha, zorder=z))
        ax.add_patch(FancyArrowPatch((cx + 0.13, cy - 0.01), (cx + 0.13, cy - 0.05),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=color, alpha=alpha, zorder=z))
        return
    ax.add_patch(FancyArrowPatch(pos[u], pos[v], connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=17, shrinkA=17,
                                 shrinkB=17, lw=width, color=color, alpha=alpha, zorder=z))


def draw_graph(ax, M, pos, labels, new=(), focus=None, k_node=None, title=""):
    n = len(M)
    ax.clear()
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    path = {(focus[0], focus[2]), (focus[2], focus[1])} if focus else set()
    for u in range(n):
        for v in range(n):
            if not M[u][v]:
                continue
            if focus and (u, v) == (focus[0], focus[1]):
                draw_edge(ax, pos, u, v, C_NEW, 3.4, z=5)
            elif (u, v) in path:
                draw_edge(ax, pos, u, v, C_PATH, 3.0, z=4)
            elif (u, v) in new:
                draw_edge(ax, pos, u, v, C_NEW, 2.0, z=3)
            else:
                draw_edge(ax, pos, u, v, C_EDGE, 1.6, alpha=0.85)
    for v in range(n):
        x, y = pos[v]
        ax.add_patch(plt.Circle((x, y), 0.13, fc=C_NODE_K if k_node == v else C_NODE,
                                ec=C_NODE_EDGE, lw=2, zorder=10))
        ax.text(x, y, labels[v], ha="center", va="center",
                fontsize=10 if len(labels[v]) < 4 else 7, fontweight="bold", zorder=11)


def draw_table(ax, M, labels, new=(), focus=None, k=0, title=""):
    n = len(M)
    ax.clear(); ax.axis("off")
    colors = [["#ffffff"] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if M[i][j]:
                colors[i][j] = C_ONE
    if k:
        for t in range(n):
            if not M[t][k - 1]:
                colors[t][k - 1] = C_LINE
            if not M[k - 1][t]:
                colors[k - 1][t] = C_LINE
    for i, j in new:
        colors[i][j] = C_NEWCELL
    tbl = ax.table(cellText=[[str(v) for v in r] for r in M], rowLabels=labels,
                   colLabels=labels, cellColours=colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15 if n <= 6 else 10)
    tbl.scale(1, 2.6 if n <= 6 else 1.8)
    if focus:
        c = tbl[(focus[0] + 1, focus[1])]
        c.set_edgecolor(C_NEW); c.set_linewidth(3); c.set_text_props(fontweight="bold")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


# =====================================================================
# 4. OUTPUT FILES
# =====================================================================
def save_animation(frames, labels, filename="warshall_animation.gif"):
    n = len(labels)
    pos = positions(n)
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[12, 1.6], hspace=0.05, wspace=0.08)
    axg, axm, axt = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    fig.suptitle("Warshall's Algorithm - Transitive Closure", fontsize=15,
                 fontweight="bold", y=0.985)

    def update(f):
        fr = frames[f]
        k, new = fr["k"], set(fr["new"])
        draw_graph(axg, fr["matrix"], pos, labels, new, fr["focus"],
                   (k - 1) if k else None, "Graph of the relation")
        draw_table(axm, fr["matrix"], labels, new, fr["focus"], k, "Matrix Mk")
        axt.clear(); axt.axis("off")
        head = fill_labels(fr["text"], labels)
        if fr["kind"] in ("intro",):
            head = f"k = {k}:  {head}"
        col = C_NEW if fr["kind"] in ("found", "final") else C_NODE_EDGE
        axt.text(0.5, 0.82, head, ha="center", va="center", fontsize=14,
                 fontweight="bold", color=col, transform=axt.transAxes)
        axt.text(0.5, 0.46, fill_labels(fr["sub"], labels), ha="center", va="center",
                 fontsize=11, color="#333333", transform=axt.transAxes)
        axt.text(0.5, 0.04, "orange edges = path i->k->j used    red = new pair    "
                 "orange node = intermediate element k", ha="center", va="bottom",
                 fontsize=8.5, color="#666666", transform=axt.transAxes)

    order = [0] * 2 + list(range(1, len(frames) - 1)) + [len(frames) - 1] * 4
    anim = FuncAnimation(fig, lambda i: update(order[i]), frames=len(order),
                         interval=1300, repeat=True)
    anim.save(filename, writer=PillowWriter(fps=1000 / 1300), dpi=90)
    plt.close(fig)
    print(f"[saved] {filename}")


def save_before_after(original, closure, labels, filename="warshall_before_after.png"):
    n = len(labels)
    pos = positions(n)
    added = {(i, j) for i in range(n) for j in range(n) if closure[i][j] and not original[i][j]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
    draw_graph(a1, original, pos, labels, title="INPUT: relation R (direct pairs)")
    draw_graph(a2, closure, pos, labels, new=added,
               title="OUTPUT: transitive closure R+ (red = added)")
    fig.suptitle("Warshall's Algorithm: Before vs After", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(filename, dpi=150); plt.close(fig)
    print(f"[saved] {filename}")


def save_steps_page(stages, labels, filename="warshall_steps.png"):
    cols = len(stages)
    fig, axes = plt.subplots(1, cols, figsize=(3.4 * cols, 4))
    axes = [axes] if cols == 1 else axes
    for ax, st in zip(axes, stages):
        draw_table(ax, st["matrix"], labels, set(st["new"]),
                   title="M0 (initial)" if st["k"] == 0 else f"After k = {st['k']}")
    fig.suptitle("Matrix after every stage (red = newly added)", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(filename, dpi=150); plt.close(fig)
    print(f"[saved] {filename}")


# =====================================================================
# 5. INPUT  (menu, set, relation)
# =====================================================================
def build_matrix(elements, pairs):
    idx = {e: i for i, e in enumerate(elements)}
    M = [[0] * len(elements) for _ in elements]
    for a, b in pairs:
        M[idx[a]][idx[b]] = 1
    return M


def ask_set():
    print("\nSTEP 1: Enter the elements of the SET A.")
    print("  Examples:   1 2 3 4        or      a, b, c, d        or      Delhi Pune Goa")
    while True:
        raw = input("Set A = ").strip().strip("{}")
        items = [t for t in re.split(r"[,\s]+", raw) if t]
        elements = list(dict.fromkeys(items))        # remove duplicates, keep order
        if len(elements) < 2:
            print("  Please enter at least 2 elements.\n")
        elif len(elements) > 10:
            print("  Please use at most 10 elements (so the picture stays readable).\n")
        else:
            print(f"  -> A = {fmt_set(elements)}   ({len(elements)} elements)")
            return elements


def parse_pairs(line):
    """Accepts '(1,2),(2,3)'   or   '1 2'   or   '1-2'   or   '1>2'."""
    found = re.findall(r"\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)", line)
    if found:
        return [(a.strip(), b.strip()) for a, b in found]
    tokens = [t for t in re.split(r"[\s,\-<>=]+", line) if t]
    if len(tokens) == 2:
        return [(tokens[0], tokens[1])]
    return None


def ask_relation(elements):
    print("\nSTEP 2: Enter the RELATION R on A  (ordered pairs).")
    print("  Type one pair per line, e.g.   1 2      or      (1,2)      or      a-b")
    print("  You can also put several in a line:   (1,2),(2,3),(3,1)")
    print("  Press ENTER on an empty line when you are finished.")
    pairs = []
    while True:
        line = input("pair> ").strip()
        if not line:
            if not pairs:
                print("  R is empty - please enter at least one pair.")
                continue
            break
        got = parse_pairs(line)
        if got is None:
            print("  Could not read that. Use e.g.  (1,2)  or  1 2")
            continue
        bad = [p for p in got if p[0] not in elements or p[1] not in elements]
        if bad:
            print(f"  {bad[0]} uses an element that is not in A = {fmt_set(elements)}. Try again.")
            continue
        for p in got:
            if p not in pairs:
                pairs.append(p)
        print("  R so far = {" + ", ".join(f"({a},{b})" for a, b in pairs) + "}")
    return pairs


def get_input():
    print("=" * 64)
    print("  WARSHALL'S ALGORITHM - TRANSITIVE CLOSURE OF A RELATION")
    print("=" * 64)
    print("\n  [ENTER]  Run the DEMO from the PPT")
    print("           A = {1, 2, 3, 4},  R = {(1,2), (2,3), (3,4), (4,1)}")
    print("  [c]      Enter YOUR OWN set and relation\n")
    choice = input("Your choice (press ENTER for demo, or type c): ").strip().lower()
    if choice == "":
        print("\nRunning the PPT demo...\n")
        return DEMO_SET[:], DEMO_REL[:]
    if choice in ("c", "custom", "2", "own", "y"):
        elements = ask_set()
        pairs = ask_relation(elements)
        return elements, pairs
    print("Not understood - running the PPT demo instead.\n")
    return DEMO_SET[:], DEMO_REL[:]


# =====================================================================
# 6. MAIN
# =====================================================================
def main():
    while True:
        elements, pairs = get_input()
        original = build_matrix(elements, pairs)
        closure, frames, stages = warshall(original)

        print_all(stages, closure, elements, original)
        save_before_after(original, closure, elements)
        save_steps_page(stages, elements)
        print("Building animation... (a few seconds)")
        save_animation(frames, elements)
        print("\nDone! Open the .gif and .png files in this folder.\n")

        again = input("Press ENTER to exit, or type r to run again: ").strip().lower()
        if again != "r":
            break
        print()


if __name__ == "__main__":
    main()