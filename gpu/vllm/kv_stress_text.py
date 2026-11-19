"""Text for the lossless KV cache's stress test (kv_stress.py), written for it: short passages in fourteen languages and scripts, and generators for the
kinds of text a tokenizer cuts into odd tokens (numbers, tables, hex, base64, code, mathematics). Our own text, under the Apache License 2.0 (LICENSE), as the rest of Glyd."""
import base64
import random

MULTILINGUAL = {
    "es": "El tren salió de la estación a las siete en punto, y los pasajeros, aún medio dormidos, buscaban sus asientos entre maletas y paraguas. Una mujer mayor repartía naranjas a los niños que viajaban solos, mientras el revisor, con paciencia, revisaba cada billete. Fuera, la niebla cubría los campos de trigo y, a lo lejos, se distinguía la silueta de un molino abandonado. Nadie hablaba de lo que había pasado la noche anterior en el pueblo, aunque todos lo tenían en la cabeza.",
    "fr": "Le marché du samedi commence avant l'aube : les maraîchers installent leurs étals, les boulangers sortent les premières miches et l'odeur du café se mêle à celle des fromages affinés. Un vieux monsieur compare longuement les poires, sous le regard amusé de la vendeuse, qui connaît ses habitudes depuis vingt ans. Vers dix heures, la place se remplit de familles, de touristes et de chiens qui tirent sur leur laisse, et personne ne semble pressé de rentrer chez soi.",
    "de": "Die Ingenieurin prüfte zum dritten Mal die Messwerte, bevor sie den Bericht an die Projektleitung schickte. Der Druck in der Leitung war über Nacht langsam gefallen, aber nicht so stark, dass die Anlage hätte abgeschaltet werden müssen. Dennoch blieb ein ungutes Gefühl, denn die Schwankungen folgten keinem erkennbaren Muster. Sie beschloss, am nächsten Morgen persönlich nach dem Ventil zu sehen, das seit dem Winter niemand mehr gewartet hatte.",
    "pt": "Depois de anos a viver na cidade grande, ela voltou para a pequena vila onde nascera, junto ao rio. A casa dos avós estava cheia de pó e de memórias: fotografias amareladas, panelas de ferro, um relógio que parara às três e meia. Passou a semana a abrir janelas, a arrancar ervas do quintal e a conversar com os vizinhos, que se lembravam dela ainda com tranças e joelhos esfolados.",
    "it": "Nel laboratorio di fisica, il professore spiegò agli studenti che un esperimento riuscito non è quello che conferma l'ipotesi, ma quello che è stato progettato in modo da poterla smentire. Gli studenti annuirono, ma uno di loro, seduto in fondo, chiese perché allora tutti i manuali raccontassero soltanto le scoperte fortunate. Il professore sorrise, posò il gesso e disse che era una domanda migliore di tutte quelle che si aspettava quel giorno.",
    "ru": "Зимой в городе темнеет рано, и уже в пять часов вечера окна домов светятся жёлтым светом. По набережной идут люди в тёплых шапках, торопясь домой, к чаю и разговорам на кухне. Старый трамвай медленно ползёт по мосту, звеня на поворотах, а внизу, подо льдом, едва слышно течёт река. Мальчик с красным рюкзаком остановился и долго смотрел на огни, пока мать не взяла его за руку.",
    "zh": "清晨的市场里，摊贩们已经摆好了新鲜的蔬菜和水果，空气中弥漫着刚出炉的包子的香味。一位老人慢慢地挑选着西红柿，一边和卖菜的姑娘聊着天气。远处传来自行车的铃声，孩子们背着书包，三三两两地走向学校。这座小城并不繁华，但每个人都熟悉彼此，日子过得平静而踏实。",
    "ja": "駅前の小さな本屋は、朝から雨の音に包まれていた。店主は古い文庫本を一冊ずつ丁寧に拭きながら、常連客が来るのを待っている。窓の外では傘をさした人々が足早に通り過ぎていくが、ここだけは時間がゆっくり流れているようだった。やがて、濡れた髪の青年が入ってきて、棚の奥から一冊の詩集を取り出した。",
    "ko": "도서관의 창가 자리에는 오후마다 같은 학생이 앉아 있었다. 그는 두꺼운 책을 펼쳐 놓고 천천히 메모를 하다가, 가끔 고개를 들어 밖에 떨어지는 은행잎을 바라보았다. 사서는 그가 무엇을 공부하는지 알지 못했지만, 폐관 시간이 되면 늘 조용히 인사를 하고 나가는 모습이 마음에 들었다.",
    "ar": "في المساء، اجتمعت العائلة حول المائدة لتناول العشاء، وكانت الجدة تحكي للأطفال قصة قديمة عن تاجر سافر عبر الصحراء بحثاً عن الرزق. استمع الصغار باهتمام، وسألها أحدهم عن الجمال التي رافقت التاجر في رحلته. ابتسمت الجدة وقالت إن الصبر هو أثمن ما يحمله المسافر معه، أكثر من الماء والذهب.",
    "hi": "गर्मियों की छुट्टियों में बच्चे नानी के गाँव जाते थे, जहाँ आम के बगीचे में दिन भर खेल चलता रहता था। शाम को सब लोग आँगन में बैठकर कहानियाँ सुनते और दूर कहीं से ढोलक की आवाज़ आती थी। रात को छत पर चारपाई बिछ जाती, और तारों को गिनते-गिनते सबकी आँख लग जाती थी।",
    "tr": "Sabah erkenden kalkan balıkçılar, ağlarını toplayıp küçük iskeleden denize açıldılar. Martılar tekneleri bir süre takip etti, sonra sahile dönüp bekleyişe geçti. Çay ocağının sahibi, her zamanki gibi ilk müşterilerini tavşankanı çayla karşıladı ve havanın öğleye doğru açacağını söyledi. Köyde herkes onun tahminlerine güvenirdi.",
    "vi": "Buổi chiều, cơn mưa rào bất chợt đổ xuống con phố nhỏ, khiến người đi đường vội vàng tìm chỗ trú. Quán cà phê đầu ngõ nhanh chóng đông khách, mùi cà phê rang lan trong không khí ẩm. Cô chủ quán vừa lau bàn vừa kể cho mọi người nghe về mùa mưa ở quê mình, khi cánh đồng ngập nước và lũ trẻ chơi đùa đến tận tối.",
    "el": "Το καλοκαίρι στο νησί κυλούσε αργά: το πρωί η θάλασσα ήταν γαλήνια, το μεσημέρι οι δρόμοι άδειαζαν από τη ζέστη και το βράδυ οι ταβέρνες γέμιζαν με φωνές και μουσική. Ο γέρος ψαράς καθόταν στην άκρη του λιμανιού και μάζευε τα δίχτυα του, χωρίς να βιάζεται, γιατί ήξερε ότι η θάλασσα δεν περιμένει κανέναν και δεν χρωστάει τίποτα σε κανέναν.",
}

CODE = {
    "rust": 'use std::collections::HashMap;\n\nfn word_counts(text: &str) -> HashMap<&str, usize> {\n    let mut counts = HashMap::new();\n    for word in text.split_whitespace() {\n        *counts.entry(word).or_insert(0) += 1;\n    }\n    counts\n}\n\nfn main() {\n    let text = "the quick brown fox jumps over the lazy dog the end";\n    let mut pairs: Vec<_> = word_counts(text).into_iter().collect();\n    pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));\n    for (word, n) in pairs {\n        println!("{word}: {n}");\n    }\n}\n',
    "go": 'package main\n\nimport (\n\t"fmt"\n\t"sync"\n)\n\nfunc worker(id int, jobs <-chan int, results chan<- int, wg *sync.WaitGroup) {\n\tdefer wg.Done()\n\tfor j := range jobs {\n\t\tresults <- j * j\n\t}\n\t_ = id\n}\n\nfunc main() {\n\tjobs := make(chan int, 100)\n\tresults := make(chan int, 100)\n\tvar wg sync.WaitGroup\n\tfor w := 0; w < 4; w++ {\n\t\twg.Add(1)\n\t\tgo worker(w, jobs, results, &wg)\n\t}\n\tfor i := 1; i <= 20; i++ {\n\t\tjobs <- i\n\t}\n\tclose(jobs)\n\twg.Wait()\n\tclose(results)\n\tfmt.Println(len(results))\n}\n',
    "sql": "SELECT c.name, COUNT(o.id) AS orders, SUM(o.total) AS revenue\nFROM customers c\nLEFT JOIN orders o ON o.customer_id = c.id AND o.created_at >= DATE '2024-01-01'\nWHERE c.country IN ('DE', 'FR', 'ES')\nGROUP BY c.name\nHAVING SUM(o.total) > 1000\nORDER BY revenue DESC\nLIMIT 25;\n\nCREATE INDEX idx_orders_customer ON orders (customer_id, created_at);\nUPDATE orders SET status = 'shipped' WHERE id IN (SELECT order_id FROM shipments WHERE shipped_at IS NOT NULL);\n",
    "html": '<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="utf-8">\n  <title>Order status</title>\n  <style>body { font-family: sans-serif; margin: 2rem; } .ok { color: #1a7f37; } .late { color: #cf222e; }</style>\n</head>\n<body>\n  <h1>Order <span id="num">#10432</span></h1>\n  <ul>\n    <li class="ok">Packed</li>\n    <li class="ok">Shipped</li>\n    <li class="late">Delayed at customs</li>\n  </ul>\n  <script>document.getElementById("num").textContent = "#" + (10432 + 1);</script>\n</body>\n</html>\n',
    "latex": "\\begin{align}\n  \\nabla \\cdot \\mathbf{E} &= \\frac{\\rho}{\\varepsilon_0}, \\\\\n  \\nabla \\times \\mathbf{B} &= \\mu_0 \\mathbf{J} + \\mu_0 \\varepsilon_0 \\frac{\\partial \\mathbf{E}}{\\partial t}, \\\\\n  \\int_{-\\infty}^{\\infty} e^{-x^2}\\,dx &= \\sqrt{\\pi}, \\qquad \\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6}.\n\\end{align}\nLet $f(x) = \\sin(x)/x$ for $x \\neq 0$ and $f(0) = 1$; then $\\lim_{x \\to 0} f(x) = 1$.\n",
}


def structured(seed, n=1):
    """Text a tokenizer cuts into many short, odd tokens: [(kind, text)] of n of each kind, from the seed."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        rows = [",".join([f"{r.randrange(0, 9999):04d}", str(r.randrange(0, 99)), f"{r.random() * 1000:.3f}", r.choice(["A", "B", "C", "NA"]), f"2024-{r.randrange(1, 13):02d}-{r.randrange(1, 29):02d}"]) for _ in range(40)]
        out.append(("csv", "id,qty,price,grade,date\n" + "\n".join(rows)))
        out.append(("csv-bare", "\n".join(rows)))
        out.append(("hex", " ".join(f"{r.randrange(0, 1 << 32):08x}" for _ in range(60))))
        out.append(("base64", base64.b64encode(bytes(r.randrange(256) for _ in range(300))).decode()))
        out.append(("digits", "".join(str(r.randrange(10)) for _ in range(300))))
        out.append(("floats", " ".join(f"{r.gauss(0, 1):.6f}" for _ in range(60))))
        out.append(("json", "[" + ", ".join('{"id": %d, "v": %.4f, "tag": "%s"}' % (r.randrange(1 << 20), r.random(), r.choice(["x", "yy", "zzz"])) for _ in range(30)) + "]"))
        out.append(("uuid", "\n".join("%08x-%04x-%04x-%04x-%012x" % (r.randrange(1 << 32), r.randrange(1 << 16), r.randrange(1 << 16), r.randrange(1 << 16), r.randrange(1 << 48)) for _ in range(25))))
        out.append(("ascii-art", "\n".join("".join(r.choice(" .:-=+*#%@") for _ in range(60)) for _ in range(25))))
        out.append(("symbols", " ".join(r.choice(["{", "}", "(", ")", "[", "]", ";", ":", ",", ".", "=>", "->", "==", "!=", "&&", "||", "<<", ">>", "::", "//", "/*", "*/", "#", "@", "$", "%", "^", "~", "`"]) for _ in range(160))))
    return out
