// Glyd's chat Markdown: what a model writes, as safe HTML. Blocks: fenced code (highlighted; a finished mermaid flowchart drawn by
// flow.js, a finished html or svg block given a Preview the page runs sandboxed), headings, quotes, tables, nested and
// numbered lists with task boxes, rules, paragraphs (a single line break kept). Inline: code, bold, italics, strikethrough, links and bare
// URLs (http and https only). Every piece of text is escaped first; nothing the model writes becomes markup of its own.
(function () {
  const esc = s => s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ---- code: one pass of a tokenizer, its classes by language family
  const KW = {
    py: "and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield None True False self match case",
    js: "async await break case catch class const continue debugger default delete do else export extends finally for from function if import in instanceof let new of return static super switch this throw try typeof var void while with yield null undefined true false",
    rs: "as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait true type unsafe use where while Some None Ok Err",
    go: "break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var nil true false",
    c: "auto break case char const continue default do double else enum extern float for goto if inline int long register return short signed sizeof static struct switch typedef union unsigned void volatile while class public private protected template typename namespace using new delete true false nullptr bool virtual override final import package extends implements interface throws throw try catch finally this super",
    sh: "if then else elif fi for while do done case esac in function return export local readonly echo cd set unset source exit",
    sql: "select from where and or not insert into values update set delete create table drop alter add join left right inner outer on group by order having limit offset as distinct union all null is like in between case when then else end primary key foreign references index",
  };
  const FAMILY = { python: "py", py: "py", javascript: "js", js: "js", jsx: "js", typescript: "js", ts: "js", tsx: "js", json: "js", rust: "rs", rs: "rs", go: "go", golang: "go",
                   c: "c", h: "c", cpp: "c", "c++": "c", cc: "c", java: "c", kotlin: "c", cs: "c", csharp: "c", swift: "c", bash: "sh", sh: "sh", shell: "sh", zsh: "sh", console: "sh", sql: "sql" };
  const SETS = {};
  for (const k in KW) SETS[k] = new Set(KW[k].split(" "));
  function highlight(code, lang) {
    const fam = FAMILY[(lang || "").toLowerCase()];
    if (!fam) return esc(code);
    const hash = fam === "py" || fam === "sh";
    const re = new RegExp([
      fam === "py" ? '"""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\'' : null,
      hash ? "#[^\\n]*" : null,
      fam === "sql" ? "--[^\\n]*" : null,
      !hash ? "//[^\\n]*|/\\*[\\s\\S]*?\\*/" : null,
      '"(?:\\\\.|[^"\\\\\\n])*"', "'(?:\\\\.|[^'\\\\\\n])*'", fam === "js" || fam === "go" ? "`[^`]*`" : null,
      "\\b\\d[\\d_.]*(?:e[+-]?\\d+)?[a-z]*\\b", "@[A-Za-z_][\\w.]*", "[A-Za-z_][\\w]*", "[\\s\\S]",
    ].filter(Boolean).join("|"), "gi");
    let out = "", m;
    while ((m = re.exec(code))) {
      const t = m[0], c0 = t[0];
      let cls = null;
      if (t.startsWith("#") && hash || t.startsWith("//") || t.startsWith("/*") || (fam === "sql" && t.startsWith("--"))) cls = "c";
      else if (c0 === '"' || c0 === "'" || c0 === "`") cls = "s";
      else if (/^\d/.test(t)) cls = "n";
      else if (c0 === "@") cls = "d";
      else if (/^[A-Za-z_]/.test(t)) {
        if (SETS[fam].has(fam === "sql" ? t.toLowerCase() : t)) cls = "k";
        else if (code[re.lastIndex] === "(") cls = "f";
        else if (/^[A-Z][a-z]/.test(t) && fam !== "sql") cls = "t";
      }
      out += cls ? `<span class="tk-${cls}">${esc(t)}</span>` : esc(t);
    }
    return out;
  }

  // ---- inline
  function inline(s) {
    const codes = [];
    s = s.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
    s = esc(s)
      .replace(/\*\*\*([^*]+)\*\*\*/g, "<b><i>$1</i></b>").replace(/\*\*([^*]+?)\*\*/g, "<b>$1</b>").replace(/__([^_]+?)__/g, "<b>$1</b>")
      .replace(/(^|[^*\w])\*([^*\n]+?)\*(?!\w)/g, "$1<i>$2</i>").replace(/(^|[^_\w])_([^_\n]+?)_(?!\w)/g, "$1<i>$2</i>")
      .replace(/~~([^~]+)~~/g, "<s>$1</s>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
      .replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?'"])/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${esc(codes[+i])}</code>`);
  }

  // ---- blocks
  const isTableSep = l => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
  const cells = l => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim());
  const LIST = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;

  function list(lines, i) {
    // a list and its nested lists, by indentation; items' continuation lines join their item
    const m0 = lines[i].match(LIST), base = m0[1].length, ordered = /\d/.test(m0[2]);
    const start = ordered ? parseInt(m0[2], 10) : 1;
    let html = ordered ? `<ol${start !== 1 ? ` start="${start}"` : ""}>` : "<ul>";
    while (i < lines.length) {
      const m = lines[i].match(LIST);
      if (!m || m[1].length < base) {
        if (!m && lines[i].trim() && /^\s+/.test(lines[i]) && html.endsWith("</li>")) { html = html.slice(0, -5) + " " + inline(lines[i].trim()) + "</li>"; i++; continue; }
        break;
      }
      if (m[1].length > base) { const [sub, j] = list(lines, i); html = html.slice(0, -5) + sub + "</li>"; i = j; continue; }
      if (/\d/.test(m[2]) !== ordered) break;
      let body = m[3], box = body.match(/^\[([ xX])\]\s+(.*)/);
      html += box ? `<li class="task"><input type="checkbox" disabled${box[1] !== " " ? " checked" : ""}> ${inline(box[2])}</li>` : `<li>${inline(body)}</li>`;
      i++;
      if (i < lines.length && !lines[i].trim() && i + 1 < lines.length && LIST.test(lines[i + 1]) && lines[i + 1].match(LIST)[1].length >= base) i++;
    }
    return [html + (ordered ? "</ol>" : "</ul>"), i];
  }

  function blocks(text) {
    const lines = text.replace(/\r/g, "").split("\n"), out = [];
    let i = 0;
    while (i < lines.length) {
      const l = lines[i];
      const fence = l.match(/^\s*(```+|~~~+)\s*([\w+#.-]*)/);
      if (fence) {
        const close = fence[1], lang = fence[2], body = [];
        i++;
        while (i < lines.length && !lines[i].trim().startsWith(close)) body.push(lines[i++]);
        const done = i < lines.length; i++;
        const src = body.join("\n"), kind = done && (/^mermaid$/i.test(lang) ? "diagram" : /^(html|svg)$/i.test(lang) || (/^xml$/i.test(lang) && /^\s*<svg/i.test(src)) ? "page" : "");
        const svg = kind === "diagram" && window.HBflow ? window.HBflow.render(src) : null;
        const tabs = svg ? '<button class="cv on" type="button" data-v="pic">Diagram</button><button class="cv" type="button" data-v="src">Code</button>'
          : kind === "page" ? '<button class="cv" type="button" data-v="run">Preview</button>' : "";
        out.push(`<div class="code${svg ? " pic" : ""}"><div class="code-bar"><span>${esc(lang || "text")}</span><span class="tools">${tabs}<button class="copy" type="button">Copy</button></span></div>` +
                 `${svg ? `<div class="diagram">${svg}</div>` : ""}<pre><code>${highlight(src, lang)}</code></pre></div>`);
        continue;
      }
      if (!l.trim()) { i++; continue; }
      const h = l.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/);
      if (h) { const n = Math.min(h[1].length + 1, 6); out.push(`<h${n}>${inline(h[2])}</h${n}>`); i++; continue; }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(l)) { out.push("<hr>"); i++; continue; }
      if (/^\s*>/.test(l)) {
        const q = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) q.push(lines[i++].replace(/^\s*>\s?/, ""));
        out.push(`<blockquote>${blocks(q.join("\n"))}</blockquote>`);
        continue;
      }
      if (l.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        const head = cells(l), align = cells(lines[i + 1]).map(c => c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : "");
        let t = "<div class=\"table\"><table><thead><tr>" + head.map((c, j) => `<th${align[j] ? ` style="text-align:${align[j]}"` : ""}>${inline(c)}</th>`).join("") + "</tr></thead><tbody>";
        i += 2;
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
          t += "<tr>" + cells(lines[i]).map((c, j) => `<td${align[j] ? ` style="text-align:${align[j]}"` : ""}>${inline(c)}</td>`).join("") + "</tr>";
          i++;
        }
        out.push(t + "</tbody></table></div>");
        continue;
      }
      if (LIST.test(l)) { const [html, j] = list(lines, i); out.push(html); i = j; continue; }
      const p = [];
      while (i < lines.length && lines[i].trim() && !/^\s*(```|~~~|#{1,6}\s|>)/.test(lines[i]) && !LIST.test(lines[i]) && !(lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]))) p.push(lines[i++]);
      out.push(`<p>${p.map(inline).join("<br>")}</p>`);
    }
    return out.join("");
  }
  window.HBmd = { render: blocks, escape: esc };
})();
