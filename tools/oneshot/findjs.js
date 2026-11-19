// The first syntax error of a page's scripts, with its line: node findjs.js FILE
const fs = require("fs"), vm = require("vm"), src = fs.readFileSync(process.argv[2], "utf8");
const m = [...src.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)];
for (const s of m) { const off = src.slice(0, s.index).split("\n").length; try { new vm.Script(s[1]); console.log("script at line", off, "parses"); } catch (e) { const ln = +(e.stack.match(/evalmachine.<anonymous>:(\d+)/) || [])[1]; console.log("script at line", off, ":", e.message, "at script line", ln); console.log(s[1].split("\n").slice(ln - 3, ln + 2).join("\n")); } }
