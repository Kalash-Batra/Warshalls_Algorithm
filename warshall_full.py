"""
=====================================================================
 Implementation and Visualization of Warshall's Algorithm
 for Transitive Closure          (Discrete Mathematics - CIA Activity)
=====================================================================

ONE FILE, NO TYPING NEEDED. Just run:

    python warshall_full.py

It will automatically:
  1. Run Warshall's algorithm on the demo graph (4-cycle V1->V2->V3->V4->V1)
  2. Print every step in the console
  3. Create  warshall_animation.gif      (full step-by-step animation)
  4. Create  warshall_before_after.png   (input graph vs transitive closure)
  5. Create  warshall_steps.png          (all stage matrices on one page)

Optional ways to run:
    python warshall_full.py --show                       # also open a live animation window
    python warshall_full.py --n 5 --edges 1-2,2-3,3-1,4-5   # your own graph
    python warshall_full.py --fast                       # faster GIF
    python warshall_full.py --input                      # type the graph interactively

Install once:   pip install matplotlib networkx pillow
"""

import argparse
import sys

import matplotlib
import networkx as nx

# --show needs a real window; otherwise use the file-only backend
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Arc, FancyArrowPatch

# ----------------------------------------------------------------------
# COLOURS
# ----------------------------------------------------------------------
C_NODE = "#a8dadc"
C_NODE_K = "#f4a261"        # the intermediate vertex vk
C_NODE_EDGE = "#1d3557"
C_EDGE = "#457b9d"          # old edge
C_PATH = "#f77f00"          # edges i->k and k->j being used right now
C_NEW = "#e63946"           # newly discovered edge
C_CELL_ONE = "#d8f3dc"
C_CELL_NEW = "#ff9aa2"
C_CELL_LINE = "#fff3b0"


# ======================================================================
# 1. THE ALGORITHM  (plus recording of every event for the animation)
# ======================================================================
def warshall_with_frames(matrix):
    """
    Runs Warshall's algorithm:
        Mk[i][j] = Mk-1[i][j]  OR  ( Mk-1[i][k]  AND  Mk-1[k][j] )
    and records animation 'frames'.  Returns (closure, frames, stages).
    """
    n = len(matrix)
    M = [row[:] for row in matrix]            # deep copy; original is preserved
    frames = []
    stages = [{"k": 0, "matrix": [r[:] for r in M], "new": []}]

    def snap(kind, k, new, focus=None, text="", sub=""):
        frames.append({
            "kind": kind, "k": k, "matrix": [r[:] for r in M],
            "new": list(new), "focus": focus, "text": text, "sub": sub,
        })

    snap("start", 0, [], text="M0 : initial adjacency matrix",
         sub="Only DIRECT edges are present.")

    for k in range(n):
        stage_new = []
        snap("intro", k + 1, stage_new,
             text=f"Stage k = {k+1} :  allow V{k+1} as an intermediate vertex",
             sub=f"Look for pairs (i, j) where  i -> V{k+1}  and  V{k+1} -> j  exist.")
        for i in range(n):
            for j in range(n):
                if not M[i][j] and M[i][k] and M[k][j]:
                    M[i][j] = 1                      # <-- the Warshall update
                    stage_new.append((i, j))
                    snap("found", k + 1, stage_new, focus=(i, j, k),
                         text=f"k = {k+1} :  NEW  V{i+1} -> V{j+1}",
                         sub=f"Because  V{i+1} -> V{k+1}  and  V{k+1} -> V{j+1}  already exist.")
        sub = ("New this stage: " + ", ".join(f"V{a+1}->V{b+1}" for a, b in stage_new)
               if stage_new else "No new paths found at this stage.")
        kind = "final" if k == n - 1 else "stage_end"
        text = (f"k = {k+1} complete  -  FINAL TRANSITIVE CLOSURE" if k == n - 1
                else f"k = {k+1} complete")
        snap(kind, k + 1, stage_new, text=text, sub=sub)
        stages.append({"k": k + 1, "matrix": [r[:] for r in M], "new": list(stage_new)})

    return M, frames, stages


# ======================================================================
# 2. CONSOLE OUTPUT
# ======================================================================
def print_matrix(M, new=(), title=""):
    n = len(M)
    if title:
        print(title)
    print("       " + "  ".join(f"V{j+1}" for j in range(n)))
    for i in range(n):
        row = ""
        for j in range(n):
            row += f" [{M[i][j]}]" if (i, j) in new else f"  {M[i][j]} "
        print(f"  V{i+1}  {row}")
    print()


def print_all(stages, closure):
    print("=" * 62)
    print("   WARSHALL'S ALGORITHM  -  STEP BY STEP")
    print("=" * 62 + "\n")
    print_matrix(stages[0]["matrix"], title="Initial Matrix M0 (direct edges only):")
    for st in stages[1:]:
        k = st["k"]
        print_matrix(st["matrix"], st["new"],
                     title=f"After k = {k}  (V{k} used as intermediate)")
        if st["new"]:
            for i, j in st["new"]:
                print(f"    NEW: V{i+1} -> V{j+1}    (via V{i+1} -> V{k} -> V{j+1})")
        else:
            print("    No new paths at this stage.")
        print("    ( [x] marks a newly added entry )\n")
    print("=" * 62)
    print_matrix(closure, title="FINAL TRANSITIVE CLOSURE (Reachability Matrix):")
    ones = sum(map(sum, closure))
    n = len(closure)
    print(f"Reachable pairs: {ones} of {n*n}")
    print("=" * 62 + "\n")


# ======================================================================
# 3. DRAWING HELPERS
# ======================================================================
def positions(n):
    return nx.circular_layout(nx.empty_graph(n), scale=1.0)


def draw_edge(ax, pos, u, v, color, width, rad=0.18, alpha=1.0, z=2):
    """Draw a directed edge u->v (or a self-loop if u == v)."""
    r = 0.13  # node radius in data units
    if u == v:
        x, y = pos[u]
        norm = (x * x + y * y) ** 0.5 or 1.0
        ox, oy = x / norm, y / norm                  # outward direction
        cx, cy = x + ox * 0.17, y + oy * 0.17
        ax.add_patch(Arc((cx, cy), 0.26, 0.26, angle=0, theta1=0, theta2=330,
                         color=color, lw=width, alpha=alpha, zorder=z))
        # small arrow head
        ax.add_patch(FancyArrowPatch((cx + 0.13, cy - 0.01), (cx + 0.13, cy - 0.05),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=color, alpha=alpha, zorder=z))
        return
    p1, p2 = pos[u], pos[v]
    ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=17,
                                 shrinkA=17, shrinkB=17, lw=width,
                                 color=color, alpha=alpha, zorder=z))


def draw_graph(ax, M, pos, new=(), focus=None, k_node=None, title=""):
    n = len(M)
    ax.clear()
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    path_edges = set()
    if focus is not None:
        i, j, k = focus
        path_edges = {(i, k), (k, j)}

    for u in range(n):
        for v in range(n):
            if not M[u][v]:
                continue
            if focus is not None and (u, v) == (focus[0], focus[1]):
                draw_edge(ax, pos, u, v, C_NEW, 3.4, z=5)
            elif (u, v) in path_edges:
                draw_edge(ax, pos, u, v, C_PATH, 3.0, z=4)
            elif (u, v) in new:
                draw_edge(ax, pos, u, v, C_NEW, 2.0, z=3)
            else:
                draw_edge(ax, pos, u, v, C_EDGE, 1.6, alpha=0.85)

    for v in range(n):
        x, y = pos[v]
        is_k = (k_node == v)
        ax.add_patch(plt.Circle((x, y), 0.13, fc=C_NODE_K if is_k else C_NODE,
                                ec=C_NODE_EDGE, lw=2, zorder=10))
        ax.text(x, y, f"V{v+1}", ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=11)


def draw_table(ax, M, new=(), focus=None, k=0, title=""):
    n = len(M)
    ax.clear()
    ax.axis("off")
    colors = [["#ffffff"] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if M[i][j]:
                colors[i][j] = C_CELL_ONE
    if k:
        for t in range(n):                            # highlight row k / column k
            if not M[t][k - 1]:
                colors[t][k - 1] = C_CELL_LINE
            if not M[k - 1][t]:
                colors[k - 1][t] = C_CELL_LINE
    for (i, j) in new:
        colors[i][j] = C_CELL_NEW
    tbl = ax.table(cellText=[[str(v) for v in r] for r in M],
                   rowLabels=[f"V{i+1}" for i in range(n)],
                   colLabels=[f"V{j+1}" for j in range(n)],
                   cellColours=colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15)
    tbl.scale(1, 2.6)
    if focus is not None:
        i, j, _ = focus
        tbl[(i + 1, j)].set_edgecolor(C_NEW)
        tbl[(i + 1, j)].set_linewidth(3)
        tbl[(i + 1, j)].set_text_props(fontweight="bold")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


# ======================================================================
# 4. OUTPUT A: THE FULL ANIMATION
# ======================================================================
def build_animation(frames, n):
    pos = positions(n)
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[12, 1.6], hspace=0.05, wspace=0.08)
    axg = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1])
    axt = fig.add_subplot(gs[1, :])
    axt.axis("off")
    fig.suptitle("Warshall's Algorithm - Transitive Closure",
                 fontsize=15, fontweight="bold", y=0.985)

    def update(f):
        fr = frames[f]
        k = fr["k"]
        k_node = (k - 1) if k else None
        new = set(fr["new"])
        draw_graph(axg, fr["matrix"], pos, new=new, focus=fr["focus"],
                   k_node=k_node, title="Reachability graph")
        draw_table(axm, fr["matrix"], new=new, focus=fr["focus"], k=k,
                   title="Matrix  Mk")
        axt.clear()
        axt.axis("off")
        col = C_NEW if fr["kind"] in ("found", "final") else "#1d3557"
        axt.text(0.5, 0.82, fr["text"], ha="center", va="center",
                 fontsize=14, fontweight="bold", color=col, transform=axt.transAxes)
        axt.text(0.5, 0.46, fr["sub"], ha="center", va="center",
                 fontsize=11, color="#333333", transform=axt.transAxes)
        # legend
        axt.text(0.5, 0.04,
                 "orange = path i->k->j used   red = new edge   orange node = intermediate vk",
                 ha="center", va="bottom", fontsize=8.5, color="#666666",
                 transform=axt.transAxes)

    return fig, update


def save_animation(frames, n, filename="warshall_animation.gif", fast=False):
    fig, update = build_animation(frames, n)
    delay_ms = 600 if fast else 1300
    # hold the first and last frame a little longer
    order = [0] * 2 + list(range(1, len(frames) - 1)) + [len(frames) - 1] * 4
    anim = FuncAnimation(fig, lambda i: update(order[i]), frames=len(order),
                         interval=delay_ms, repeat=True)
    anim.save(filename, writer=PillowWriter(fps=1000 / delay_ms), dpi=90)
    plt.close(fig)
    print(f"[saved] {filename}  ({len(order)} frames)")


def show_live(frames, n, fast=False):
    fig, update = build_animation(frames, n)
    delay_ms = 600 if fast else 1300
    anim = FuncAnimation(fig, lambda i: update(i), frames=len(frames),
                         interval=delay_ms, repeat=True)
    plt.show()
    return anim


# ======================================================================
# 5. OUTPUT B: BEFORE vs AFTER PICTURE
# ======================================================================
def save_before_after(original, closure, filename="warshall_before_after.png"):
    n = len(original)
    pos = positions(n)
    added = {(i, j) for i in range(n) for j in range(n)
             if closure[i][j] and not original[i][j]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
    draw_graph(a1, original, pos, title="INPUT graph (direct edges only)")
    draw_graph(a2, closure, pos, new=added,
               title="TRANSITIVE CLOSURE (red = discovered by Warshall)")
    fig.suptitle("Warshall's Algorithm: Before vs After", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[saved] {filename}")


# ======================================================================
# 6. OUTPUT C: ALL STAGES ON ONE PAGE (good for the report)
# ======================================================================
def save_steps_page(stages, filename="warshall_steps.png"):
    n = len(stages[0]["matrix"])
    cols = len(stages)
    fig, axes = plt.subplots(1, cols, figsize=(3.4 * cols, 4))
    if cols == 1:
        axes = [axes]
    for ax, st in zip(axes, stages):
        title = "M0 (initial)" if st["k"] == 0 else f"After k = {st['k']}"
        draw_table(ax, st["matrix"], new=set(st["new"]), title=title)
    fig.suptitle("Warshall's Algorithm - matrix after every stage (red = new)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[saved] {filename}")


# ======================================================================
# 7. GRAPH INPUT
# ======================================================================
DEMO = [[0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]]


def matrix_from_edges(n, edge_string):
    M = [[0] * n for _ in range(n)]
    for part in edge_string.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.replace(" ", "").split("-")
        a, b = int(a), int(b)
        if not (1 <= a <= n and 1 <= b <= n):
            sys.exit(f"Edge {a}-{b}: vertex numbers must be between 1 and {n}")
        M[a - 1][b - 1] = 1
    return M


def read_interactively():
    n = int(input("Number of vertices: "))
    M = [[0] * n for _ in range(n)]
    print("Enter edges as 'from to' (1-based). Blank line to finish.")
    while True:
        line = input("edge> ").strip()
        if not line:
            break
        a, b = map(int, line.split())
        if 1 <= a <= n and 1 <= b <= n:
            M[a - 1][b - 1] = 1
        else:
            print(f"  numbers must be 1..{n}")
    return M


# ======================================================================
# 8. MAIN
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Warshall's algorithm with full animation")
    ap.add_argument("--n", type=int, help="number of vertices (use with --edges)")
    ap.add_argument("--edges", type=str, help='edges like "1-2,2-3,3-1"')
    ap.add_argument("--input", action="store_true", help="type the graph interactively")
    ap.add_argument("--show", action="store_true", help="open a live animation window")
    ap.add_argument("--fast", action="store_true", help="faster animation")
    ap.add_argument("--no-gif", action="store_true", help="skip creating the GIF")
    args = ap.parse_args()

    if args.input:
        graph = read_interactively()
    elif args.n and args.edges:
        graph = matrix_from_edges(args.n, args.edges)
    else:
        graph = DEMO
        print("(Using the demo graph: V1->V2->V3->V4->V1)\n")

    closure, frames, stages = warshall_with_frames(graph)

    print_all(stages, closure)

    save_before_after(graph, closure)
    save_steps_page(stages)
    if not args.no_gif:
        print("Building animation... (takes a few seconds)")
        save_animation(frames, len(graph), fast=args.fast)
    if args.show:
        show_live(frames, len(graph), fast=args.fast)

    print("\nDone! Open the .gif and .png files in the same folder.")


if __name__ == "__main__":
    main()
