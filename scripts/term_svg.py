"""Terminal captures as one SVG image, text as captured: panels side by
side, each a title, the command line and the command's output.

    python scripts/term_svg.py [--stack] OUT.svg "TITLE|COMMAND|FILE[|LINES]" ...

--stack: panels one above the other (else side by side).

LINES (optional): first-last of the file's lines to show (1-based).
Lines containing a `!` marker in LINES as `first-last!pattern` are drawn
bright: every line holding `pattern` (e.g. a GPU's memory row)."""
import html, sys

W_CH, H_LN, PAD, GAP, TOP = 7.8, 17, 16, 20, 34  # a character's width, a line's height, margins, between panels, title bar


def panel(spec):
    title, command, path, *rest = spec.split("|")
    lines = open(path).read().rstrip("\n").split("\n")
    bright = None
    if rest:
        span, _, bright = rest[0].partition("!")
        a, b = (int(x) for x in span.split("-"))
        lines = lines[a - 1 : b]
    return title, command, [l.rstrip() for l in lines], bright


def main():
    stack = sys.argv[1] == "--stack"
    args = sys.argv[2:] if stack else sys.argv[1:]
    out, specs = args[0], [panel(s) for s in args[1:]]
    widths = [max(len(c) + 2, *(len(l) for l in ls)) * W_CH + 2 * PAD for _, c, ls, _ in specs]
    heights = [TOP + PAD + (len(ls) + 1) * H_LN + PAD for _, _, ls, _ in specs]
    if stack:
        widths = [max(widths)] * len(widths)
        width, height = widths[0], sum(heights) + GAP * (len(specs) - 1)
    else:
        heights = [max(heights)] * len(heights)
        width, height = sum(widths) + GAP * (len(specs) - 1), heights[0]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="13">']
    x = y0 = 0.0
    for (title, command, lines, bright), w, h in zip(specs, widths, heights):
        svg.append(f'<rect x="{x:.0f}" y="{y0:.0f}" width="{w:.0f}" height="{h:.0f}" rx="8" fill="#0d1117"/>')
        svg.append(f'<rect x="{x:.0f}" y="{y0:.0f}" width="{w:.0f}" height="{TOP}" rx="8" fill="#21262d"/><rect x="{x:.0f}" y="{y0 + TOP - 8:.0f}" width="{w:.0f}" height="8" fill="#21262d"/>')
        for i, c in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
            svg.append(f'<circle cx="{x + 16 + 18 * i:.0f}" cy="{y0 + TOP / 2:.0f}" r="6" fill="{c}"/>')
        svg.append(f'<text x="{x + w / 2:.0f}" y="{y0 + TOP / 2 + 5:.0f}" fill="#c9d1d9" text-anchor="middle" font-weight="bold">{html.escape(title)}</text>')
        y = y0 + TOP + PAD + 12
        svg.append(f'<text x="{x + PAD:.0f}" y="{y:.0f}" fill="#7ee787" xml:space="preserve">$ {html.escape(command)}</text>')
        for l in lines:
            y += H_LN
            color = "#f0f6fc" if bright and bright in l else "#8b949e"
            weight = ' font-weight="bold"' if bright and bright in l else ""
            svg.append(f'<text x="{x + PAD:.0f}" y="{y:.0f}" fill="{color}"{weight} xml:space="preserve">{html.escape(l)}</text>')
        if stack:
            y0 += h + GAP
        else:
            x += w + GAP
    svg.append("</svg>")
    open(out, "w").write("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
