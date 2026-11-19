// Screenshots side by side in one image: node sheet.js OUT.png A.png B.png ... (2 columns, scaled to half size)
const { chromium } = require("/Users/suryakoritala/sigbound-forge/node_modules/playwright-core"), fs = require("fs");
const [out, ...ims] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch({ channel: "chrome", headless: true }), p = await b.newPage({ viewport: { width: 1280, height: 360 * Math.ceil(ims.length / 2) } });
  await p.setContent(`<body style="margin:0;display:grid;grid-template-columns:640px 640px;background:#000">${ims.map(f => `<img style="width:640px;height:360px" src="data:image/png;base64,${fs.readFileSync(f).toString("base64")}">`).join("")}</body>`);
  await p.screenshot({ path: out }); await b.close();
})();
