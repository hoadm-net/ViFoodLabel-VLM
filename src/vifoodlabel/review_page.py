"""Offline review page for the taxonomy-validation human pass.

`build_review_page(rows, definitions)` returns one self-contained HTML file
(no network, no server: opened straight from disk) that shows each sampled
error next to its label photograph, pre-selects the judge's category, and
exports the reviewer's decisions as a CSV that
`taxonomy-validation --import-review` merges back. Progress is kept in the
browser's localStorage so a review can span several sittings.

The rule-based category is deliberately NOT included in `rows`, and the
caller shuffles them: the human is confirming the judge's label as the
reference, and seeing the rule's answer (or rows grouped by it) would anchor
them toward agreeing with the rule being validated.
"""

from __future__ import annotations

import html
import json

REVIEW_COLUMNS = ["image_id", "model_key", "field", "human_category", "human_confirmed", "human_note"]

_TEMPLATE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Taxonomy review</title>
<style>
:root { --bg:#fafafa; --fg:#1a1a1a; --muted:#666; --card:#fff; --line:#ddd; --accent:#0a66c2; --ok:#1b7f3b; }
@media (prefers-color-scheme: dark) { :root { --bg:#161616; --fg:#eee; --muted:#aaa; --card:#222; --line:#3a3a3a; --accent:#5aa0f0; --ok:#4cc47a; } }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
header { position:sticky; top:0; z-index:2; background:var(--card); border-bottom:1px solid var(--line); padding:8px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
header b { font-size:15px; }
button { font:inherit; padding:5px 12px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--fg); cursor:pointer; }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
.wrap { display:grid; grid-template-columns:minmax(0,1.1fr) minmax(0,1fr); gap:16px; padding:16px; }
@media (max-width:900px) { .wrap { grid-template-columns:1fr; } }
.img { background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:auto; max-height:calc(100vh - 90px); }
.img img { display:block; width:100%; cursor:zoom-in; }
.img img.zoom { width:220%; cursor:zoom-out; }
.panel { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
.meta { color:var(--muted); margin-bottom:8px; }
.box { white-space:pre-wrap; border:1px solid var(--line); border-radius:6px; padding:8px; margin:4px 0 10px; max-height:28vh; overflow:auto; font-size:13px; }
label { font-weight:600; display:block; margin-top:6px; }
select, textarea { font:inherit; width:100%; padding:6px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--fg); }
textarea { height:48px; }
.status { font-weight:600; }
.status.done { color:var(--ok); }
details { margin-top:10px; } details pre { white-space:pre-wrap; font-size:12px; }
kbd { border:1px solid var(--line); border-radius:4px; padding:0 4px; font-size:12px; }
</style>
</head>
<body>
<header>
  <b>Taxonomy review</b>
  <span id="pos"></span>
  <span id="count" class="status"></span>
  <button id="prev">&larr; Trước</button>
  <button id="confirm" class="primary">Xác nhận &amp; tiếp &rarr;</button>
  <button id="next">Bỏ qua &rarr;</button>
  <button id="nextTodo">Tới dòng chưa duyệt</button>
  <button id="export">Xuất CSV</button>
  <span class="meta"><kbd>Enter</kbd> xác nhận · <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> di chuyển · bấm ảnh để phóng to</span>
</header>
<div class="wrap">
  <div class="img"><img id="photo" alt="label photo"></div>
  <div class="panel">
    <div class="meta" id="meta"></div>
    <label>Ground truth (nhãn chuẩn)</label><div class="box" id="gt"></div>
    <label>Model dự đoán</label><div class="box" id="pred"></div>
    <div class="meta" id="flags"></div>
    <label>Gợi ý của judge</label><div class="box" id="judge"></div>
    <label for="cat">Loại lỗi đúng (human_category)</label>
    <select id="cat"></select>
    <label for="note">Ghi chú (tuỳ chọn)</label>
    <textarea id="note"></textarea>
    <div class="status" id="state"></div>
    <details><summary>Định nghĩa các loại lỗi</summary><pre id="defs"></pre></details>
  </div>
</div>
<script>
const ROWS = __ROWS__;
const CATEGORIES = __CATEGORIES__;
const DEFS = __DEFS__;
const KEY = "taxonomy-review-v1";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { saved = {}; }
const rid = r => r.image_id + "|" + r.model_key + "|" + r.field;
let i = 0;
const $ = id => document.getElementById(id);
$("cat").innerHTML = CATEGORIES.map(c => `<option value="${c}">${c}</option>`).join("");
$("defs").textContent = DEFS;
function persist() { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} }
function doneCount() { return ROWS.filter(r => saved[rid(r)] && saved[rid(r)].confirmed).length; }
function render() {
  const r = ROWS[i], s = saved[rid(r)] || {};
  $("photo").src = "images/" + r.image_id + ".jpeg";
  $("photo").classList.remove("zoom");
  $("pos").textContent = `Dòng ${i + 1} / ${ROWS.length}`;
  $("count").textContent = `Đã duyệt: ${doneCount()} / ${ROWS.length}`;
  $("count").className = "status" + (doneCount() === ROWS.length ? " done" : "");
  $("meta").textContent = `Ảnh ${r.image_id} · model ${r.model_key} · trường "${r.field}"`;
  $("gt").textContent = r.gt_value || "(trống)";
  $("pred").textContent = r.pred_value || "(trống)";
  $("flags").textContent = "Cờ phản hồi: " + r.flags;
  $("judge").textContent = `${r.judge_category || "(không có)"} — ${r.judge_rationale || ""}`;
  $("cat").value = s.category || r.judge_category || CATEGORIES[0];
  $("note").value = s.note || "";
  $("state").textContent = s.confirmed ? "✓ Đã xác nhận" : "Chưa duyệt";
  $("state").className = "status" + (s.confirmed ? " done" : "");
}
function store(confirmed) {
  const r = ROWS[i], k = rid(r), prev = saved[k] || {};
  saved[k] = { category: $("cat").value, note: $("note").value, confirmed: confirmed || !!prev.confirmed };
  persist();
}
function go(d) { store(false); i = Math.min(ROWS.length - 1, Math.max(0, i + d)); render(); }
$("prev").onclick = () => go(-1);
$("next").onclick = () => go(1);
$("confirm").onclick = () => { store(true); if (i < ROWS.length - 1) i++; render(); };
$("nextTodo").onclick = () => { store(false); const j = ROWS.findIndex(r => !(saved[rid(r)] && saved[rid(r)].confirmed)); if (j >= 0) { i = j; render(); } };
$("photo").onclick = () => $("photo").classList.toggle("zoom");
$("cat").onchange = () => store(false);
$("note").oninput = () => store(false);
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  if (e.key === "Enter") $("confirm").click();
  if (e.key === "ArrowRight") go(1);
  if (e.key === "ArrowLeft") go(-1);
});
function csvCell(v) { v = String(v ?? ""); return /[",\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
$("export").onclick = () => {
  store(false);
  const head = __COLUMNS__;
  const lines = [head.join(",")].concat(ROWS.map(r => {
    const s = saved[rid(r)] || {};
    return [r.image_id, r.model_key, r.field, s.category || r.judge_category || "", s.confirmed ? 1 : 0, s.note || ""].map(csvCell).join(",");
  }));
  const blob = new Blob(["\\ufeff" + lines.join("\\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "taxonomy_review_done.csv";
  a.click();
};
render();
</script>
</body>
</html>
"""


def _js(value) -> str:
    # </script> inside a JSON string would end the script block early.
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def build_review_page(rows: list[dict], categories: list[str], definitions: str) -> str:
    return (
        _TEMPLATE.replace("__ROWS__", _js(rows))
        .replace("__CATEGORIES__", _js(categories))
        .replace("__DEFS__", _js(definitions))
        .replace("__COLUMNS__", _js(REVIEW_COLUMNS))
    )


def category_definitions(judge_prompt: str) -> str:
    """The category list as given to the judge, so the human labels against
    the exact same definitions."""
    start = judge_prompt.find("Error categories")
    end = judge_prompt.find("If several apply")
    return html.unescape(judge_prompt[start:end].strip()) if start >= 0 and end > start else judge_prompt
