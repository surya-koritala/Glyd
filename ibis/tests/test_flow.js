// flow.js on its own: parsing, every shape, back edges, escaping, and refusing what it does not draw.   node tests/test_flow.js
const F = require("../tools/web/flow.js"), assert = require("assert");
const g = F.parse(`flowchart TD
  Start([Start]) --> A[Enter <b>creds</b>] --> V{Valid?}
  V -->|No| E[Show error] --> A
  V -- Yes --> OK((Done))
  A & V -.-> L[(Log)]
  classDef x fill:#f00
  H{{hex}} ==> P[/para/] --- S[[sub]] --> F>flag]`);
assert.deepStrictEqual(g.order, ["Start", "A", "V", "E", "OK", "L", "H", "P", "S", "F"]);
assert.strictEqual(g.nodes.V.shape, "diamond"); assert.strictEqual(g.nodes.L.shape, "db"); assert.strictEqual(g.nodes.OK.shape, "circle");
assert.strictEqual(g.edges.length, 10);
assert.deepStrictEqual(g.edges.filter(e => e.label).map(e => e.label), ["No", "Yes"]);
assert(g.edges.find(e => e.a === "V" && e.b === "L").dot && g.edges.find(e => e.a === "H").thick && !g.edges.find(e => e.a === "P").arrow);
const svg = F.render(`graph LR\n  a["<script>x</script>"] --> b`);
assert(svg.startsWith("<svg") && !svg.includes("<script>") && svg.includes("&lt;script&gt;"));
assert(F.render(`flowchart TD\n A-->B\n B-->A`).includes("<path"));
assert.strictEqual(F.render("sequenceDiagram\n A->>B: hi"), null);
console.log("flow ok");
