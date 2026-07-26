"""Prompt construction for the {vi,en} x {zero,one}-shot conditions.

The one-shot condition demonstrates the expected JSON *format* with a
synthetic, entirely made-up product (fake brand, fake numbers) described in
text only — no real benchmark image is ever used as the exemplar, so all 600
images stay eligible for scoring under every condition (no data leakage).
"""

from __future__ import annotations

from typing import Literal

Language = Literal["vi", "en"]
Shot = Literal["zero", "one"]

FIELD_DESCRIPTIONS_VI = """\
- product_name (chuỗi): tên đầy đủ của sản phẩm như in trên nhãn.
- ingredient (mảng chuỗi): danh sách thành phần, mỗi phần tử là một dòng/mục thành phần (giữ nguyên tỉ lệ % nếu có ghi trên nhãn).
- additive (mảng chuỗi): danh sách phụ gia thực phẩm được liệt kê riêng trên nhãn (chất bảo quản, chất tạo màu, chất nhũ hóa, hương liệu...).
- warning (mảng chuỗi): các cảnh báo/khuyến cáo an toàn thực phẩm in trên nhãn (dị ứng, bảo quản, đối tượng không nên dùng...).
- nutrition (mảng đối tượng {name, value}): bảng thông tin dinh dưỡng, mỗi phần tử gồm tên chỉ tiêu (name) và giá trị kèm đơn vị (value). PHẢI ghép đúng tên với giá trị tương ứng của nó trên bảng, kể cả khi bảng có song ngữ Việt-Anh hoặc nhiều dòng.
- origin (chuỗi): nước/nơi sản xuất hoặc xuất xứ.
- net_weight (chuỗi): khối lượng tịnh/thể tích thực.
- mfg_date (chuỗi): ngày sản xuất, hoặc hướng dẫn cách xem ngày sản xuất nếu nhãn không in trực tiếp ngày cụ thể.
- expiry_date (chuỗi): hạn sử dụng, hoặc hướng dẫn cách xem hạn sử dụng nếu nhãn không in trực tiếp ngày cụ thể."""

FIELD_DESCRIPTIONS_EN = """\
- product_name (string): the full product name as printed on the label.
- ingredient (array of strings): the ingredient list, one entry per ingredient line/item (keep the percentage if printed).
- additive (array of strings): food additives listed separately on the label (preservatives, colorants, emulsifiers, flavorings...).
- warning (array of strings): safety warnings/advisories printed on the label (allergens, storage instructions, who should avoid it...).
- nutrition (array of {name, value} objects): the nutrition facts table; each entry has the nutrient name (name) and its value with unit (value). You MUST pair each name with its correct corresponding value from the table, even when the table is bilingual Vietnamese-English or spans multiple rows.
- origin (string): country/place of manufacture or origin.
- net_weight (string): net weight/volume.
- mfg_date (string): manufacturing date, or instructions for finding it if not printed directly.
- expiry_date (string): expiry date, or instructions for finding it if not printed directly."""

TASK_INSTRUCTION_VI = """\
Bạn là hệ thống trích xuất thông tin có cấu trúc từ ảnh nhãn thực phẩm tiếng Việt.
Hãy đọc kỹ ảnh nhãn sản phẩm được đính kèm và trích xuất thông tin thành một đối tượng JSON DUY NHẤT với đúng 9 trường sau (giữ nguyên tên trường bằng tiếng Anh, giữ nguyên ngôn ngữ/nội dung gốc trên nhãn cho các giá trị):

{fields}

Quy tắc bắt buộc:
1. Chỉ xuất ra JSON hợp lệ, không kèm giải thích, không dùng markdown code fence.
2. Nếu một trường không xuất hiện trên nhãn, dùng chuỗi rỗng "" (với trường dạng chuỗi) hoặc mảng rỗng [] (với trường dạng mảng).
3. Không tự suy đoán hay bịa thông tin không có trên ảnh.
4. Giữ nguyên dấu tiếng Việt và chính tả như trên nhãn.{example}"""

TASK_INSTRUCTION_EN = """\
You are a structured information extraction system for Vietnamese food product label images.
Carefully read the attached label image and extract the information into a SINGLE JSON object with exactly these 9 fields (keep the field names in English; keep the values in whatever language/content actually appears on the label):

{fields}

Mandatory rules:
1. Output valid JSON only — no explanation, no markdown code fence.
2. If a field does not appear on the label, use an empty string "" (for string fields) or an empty array [] (for array fields).
3. Do not guess or hallucinate information that is not visible on the image.
4. Preserve Vietnamese diacritics and spelling exactly as printed on the label.{example}"""

_SYNTHETIC_EXAMPLE_VI = """


Ví dụ minh họa định dạng đầu ra mong muốn (đây là sản phẩm hư cấu, không liên quan tới ảnh thực tế bên dưới, chỉ để minh họa cấu trúc JSON và cách ghép cặp tên-giá trị dinh dưỡng):
Giả sử nhãn ghi: "Kẹo dừa sáp Cô Ba, Thành phần: Dừa sáp (60%), Đường, Sữa đặc. Phụ gia: Chất bảo quản Kali sorbat (E202). Cảnh báo: Có chứa sữa. Bảng dinh dưỡng (trên 100g) - Năng lượng/Energy: 420kcal, Chất béo/Fat: 18g, Natri/Sodium: 45mg. Xuất xứ: Việt Nam. Khối lượng tịnh: 200g. NSX: 01/01/2026. HSD: 01/01/2027."
=>
{{"product_name": "Kẹo dừa sáp Cô Ba", "ingredient": ["Dừa sáp (60%)", "Đường", "Sữa đặc"], "additive": ["Chất bảo quản Kali sorbat (E202)"], "warning": ["Có chứa sữa."], "nutrition": [{{"name": "Năng lượng", "value": "420 kcal"}}, {{"name": "Chất béo", "value": "18 g"}}, {{"name": "Natri", "value": "45 mg"}}], "origin": "Việt Nam", "net_weight": "200g", "mfg_date": "01/01/2026", "expiry_date": "01/01/2027"}}

Bây giờ hãy trích xuất thông tin từ ảnh nhãn thực tế bên dưới theo đúng định dạng trên."""

_SYNTHETIC_EXAMPLE_EN = """


Illustrative example of the expected output format (this is a fictional product unrelated to the actual image below, shown only to demonstrate the JSON structure and nutrition name-value pairing):
Suppose the label reads: "Co Ba Wax Coconut Candy, Ingredients: Wax coconut (60%), Sugar, Condensed milk. Additives: Potassium sorbate preservative (E202). Warning: Contains milk. Nutrition facts (per 100g) - Energy: 420kcal, Fat: 18g, Sodium: 45mg. Origin: Vietnam. Net weight: 200g. MFG: 01/01/2026. EXP: 01/01/2027."
=>
{{"product_name": "Co Ba Wax Coconut Candy", "ingredient": ["Wax coconut (60%)", "Sugar", "Condensed milk"], "additive": ["Potassium sorbate preservative (E202)"], "warning": ["Contains milk."], "nutrition": [{{"name": "Energy", "value": "420 kcal"}}, {{"name": "Fat", "value": "18 g"}}, {{"name": "Sodium", "value": "45 mg"}}], "origin": "Vietnam", "net_weight": "200g", "mfg_date": "01/01/2026", "expiry_date": "01/01/2027"}}

Now extract the information from the actual label image below, following the same format."""


def condition_name(language: Language, shot: Shot) -> str:
    return f"{language}_{shot}"


def build_instruction(language: Language, shot: Shot) -> str:
    example = ""
    if shot == "one":
        example = _SYNTHETIC_EXAMPLE_VI if language == "vi" else _SYNTHETIC_EXAMPLE_EN
    if language == "vi":
        return TASK_INSTRUCTION_VI.format(fields=FIELD_DESCRIPTIONS_VI, example=example)
    return TASK_INSTRUCTION_EN.format(fields=FIELD_DESCRIPTIONS_EN, example=example)


# The canonical Tier-1 benchmark condition.
CANONICAL_LANGUAGE: Language = "vi"
CANONICAL_SHOT: Shot = "zero"
CANONICAL_CONDITION = condition_name(CANONICAL_LANGUAGE, CANONICAL_SHOT)

ALL_PROMPT_CONDITIONS: list[tuple[Language, Shot]] = [
    ("vi", "zero"),
    ("vi", "one"),
    ("en", "zero"),
    ("en", "one"),
]
