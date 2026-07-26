# Ground truth annotation guidelines

The guidelines human annotators follow when correcting the auto-generated
draft JSON for each image against the [9-field schema](dataset.md). The two
fields annotators get wrong most often — the `ingredient`/`additive` split
and the `warning` definition — get the most detail below; §3 covers smaller
notes for the remaining fields.

The classification rules here (§1, §2) are also distilled directly into the
model prompts (`prompts/classification_rules_{vi,en}.txt`) — a zero-shot
model can't be expected to reconstruct an idiosyncratic labeling convention
it was never told, so the prompt states the same rules used to produce
ground truth. This keeps the benchmark measuring extraction ability given a
well-specified task, not "did the model guess our convention."

## 1. `ingredient` vs `additive`

### 1.1 Functional principle

Additives are technically part of the "Thành phần" (ingredients) list on the
label too — they can't be split by *position*, only by **function**.

An item is `additive` if it satisfies at least one of:

| # | Condition | Examples |
|---|---|---|
| (a) | Comes with a functional additive group name | chất bảo quản (preservative), chất tạo ngọt (sweetener), chất điều vị (flavor enhancer), chất nhũ hóa (emulsifier), chất ổn định (stabilizer), chất chống oxy hóa (antioxidant), chất tạo màu (colorant), chất chống đông vón (anti-caking), chất giữ ẩm (humectant), chất tạo phức kim loại (sequestrant), chất xử lý bột (flour treatment agent), enzyme/men xử lý công nghiệp (industrial processing enzyme) |
| (b) | Has an INS/E-number code | "211", "E330", "INS 621" — with or without a name |
| (c) | Is a flavoring (any kind) | synthetic / natural / nature-identical — flavoring is ALWAYS additive |

Otherwise it's `ingredient`: base materials contributing mass/structure/
nutrition (flour, plain sugar, oil/fat, milk, eggs, fruit, meat/fish, salt,
raw spices, water), and fortification vitamins/minerals (per Codex
convention, fortification nutrients don't count as additives even in small
amounts).

### 1.2 Resolved edge cases (apply consistently, don't re-litigate)

| Item | Classification | Reason |
|---|---|---|
| Bột ngọt / mì chính (MSG) | `additive` | Flavor enhancer, INS 621 |
| Tinh bột biến tính (modified starch) | `additive` | Processing/stabilizing agent |
| Tinh bột thường (corn, cassava) | `ingredient` | Base material |
| Men vi sinh / probiotic | `ingredient` | Not an additive by regulation |
| Vitamin/mineral fortification | `ingredient` | Nutrient, not an additive |
| Flavoring (any kind) | `additive` | Odor-creating = technological function |
| Natural extract used for coloring (e.g. "chiết xuất cà rốt tím") | `additive` if the label states a coloring purpose; otherwise `ingredient` | Classify by the function stated on the label — don't guess |

### 1.3 Overlap rule — two cases to distinguish

**Case 1 — an additive is NESTED inside another ingredient's parentheses**:
keep the entire declared string as-is in `ingredient[]` (don't split the
parentheses — stay faithful to the original label), **and also** extract
that additive separately into `additive[]`. This is intentional duplication,
not an error — don't remove it when reviewing.

```
Label: "Chất béo thực vật (chứa chất chống oxy hóa: dl-alpha Tocopherol)"
→ ingredient: ["Chất béo thực vật (chứa chất chống oxy hóa: dl-alpha Tocopherol)"]
→ additive:   ["dl-alpha Tocopherol"]
```

**Case 2 — an additive is an INDEPENDENT top-level item** (standing
alongside other ingredients, separated by a top-level comma): it goes only
into `additive[]`, not `ingredient[]`.

```
Label: "Bột mì, Đường, Chất nhũ hóa: Lecithin đậu nành, Hương liệu tổng hợp"
→ ingredient: ["Bột mì", "Đường"]
→ additive:   ["Chất nhũ hóa: Lecithin đậu nành", "Hương liệu tổng hợp"]
```

Quick review check: a valid Case 1 never produces two *identical* strings
across `ingredient`/`additive` — the `ingredient` string is always longer
(it contains the additive nested inside). Two fields with an exact-match
string is a Case 2 error — fix by removing that string from `ingredient[]`.

### 1.4 Full worked example (both cases combined)

```
Label: "Bột mì, Đường, Chất béo thực vật (chứa chất chống oxy hóa: dl-alpha
Tocopherol), Bột cacao, Chất nhũ hóa: Lecithin đậu nành, Hương liệu tổng hợp
(cà phê trắng)"
```

```json
{
  "ingredient": [
    "Bột mì",
    "Đường",
    "Chất béo thực vật (chứa chất chống oxy hóa: dl-alpha Tocopherol)",
    "Bột cacao"
  ],
  "additive": [
    "dl-alpha Tocopherol",
    "Lecithin đậu nành",
    "Hương liệu tổng hợp (cà phê trắng)"
  ]
}
```

- `dl-alpha Tocopherol` is nested inside "Chất béo thực vật"'s parentheses →
  Case 1: kept in `ingredient` as part of the full string, and also split
  out into `additive`.
- `Chất nhũ hóa: Lecithin đậu nành` and `Hương liệu tổng hợp` are
  independent top-level items → Case 2: `additive` only.

### 1.5 Watch out for fabricated content

The draft sometimes contains an `ingredient[]` item that doesn't match any
text actually visible on the image — usually the pre-annotation system
paraphrasing/substituting different wording.

Real case observed: the label read *"hương liệu giống tự nhiên và tự nhiên
dùng cho thực phẩm"*, but the draft wrote *"chất xơ tự nhiên và tự nhiên
dùng cho thực phẩm"* in `ingredient[]` — that phrase doesn't exist on the
label.

When reviewing: if an `ingredient` entry reads oddly and you can't find
matching text on the image, it's likely this failure mode — correct it to
match the label's actual printed text (or delete if there's no basis for it).

## 2. `warning`

### 2.1 Priority order when evaluating a sentence

**Step 1 — check for an explicit label-declared header.** If the label
itself titles a section as a warning for that sentence (under "Thông tin
cảnh báo:", "Cảnh báo:") → it always counts as `warning`, regardless of
whether the content passes the functional test in Step 2.

> Example: "Bên trong có khí Nitơ để bảo quản." under header "Thông tin
> cảnh báo:" → still `warning`, even though the sentence itself doesn't
> state a specific health risk — because the manufacturer self-classified
> it as a warning.

**Step 2 — if there's NO explicit warning header** (no header, a generic
header like "Lưu ý:", or the header is usage/storage instructions) → apply
the functional test: the sentence is `warning` **if and only if** it
satisfies both:

- (a) states a specific health/safety risk when consuming or misusing the product;
- (b) omitting that information could lead to a **direct** health
  consequence for the user (not just reduced product quality/taste).

Satisfying only (a) or only (b) → not a warning.

*(Rationale for Step 1: respect the manufacturer's own classification when
they've stated it explicitly, while still applying an objective standard
(Step 2) when the label isn't explicit — avoiding inconsistency across
labels that word the same kind of content differently.)*

### 2.2 Full classification table (functional test — Step 2)

| Category | Example | `warning`? |
|---|---|---|
| Allergens | "Có chứa sữa, đậu nành", "Có thể chứa trứng, đậu phộng" | ✅ Yes |
| Contraindicated group / age limit (categorical) | "Không dùng cho trẻ dưới 3 tuổi", "Dành cho người từ 15 tuổi trở lên" | ✅ Yes |
| Mandatory legal disclaimer (supplements) | "Sản phẩm này không phải là thuốc, không có tác dụng thay thế thuốc chữa bệnh" | ✅ Yes |
| Physical hazard (choking, sharp edges) | "Cẩn thận khi cho trẻ nhỏ ăn", sharp-can-rim warnings | ✅ Yes |
| Overdose with a stated health consequence | "Dùng quá 20g/ngày có thể gây nhuận tràng" | ✅ Yes |
| Safety when spoiled/expired | "Không sử dụng sản phẩm đã hết hạn hoặc có dấu hiệu mốc, hư hỏng." | ✅ Yes |
| Dosage by age bracket (not a restriction) | "Dưới 10 tuổi: 2 ly/ngày.", "Trên 10 tuổi: 3 ly đến 1 hộp/ngày." | ❌ No — this is a dosage table, not an age restriction |
| Sentence only stating where/how the mfg/exp date is printed | "Ngày sản xuất (NSX): được in trên bao bì.", "NSX: 08 tháng trước HSD." | ❌ No — belongs to `mfg_date`/`expiry_date`, don't duplicate here |
| Storage instructions | "Bảo quản nơi khô ráo, thoáng mát, tránh ánh nắng trực tiếp" | ❌ No |
| Usage instructions | "Lắc đều trước khi dùng", "Pha với nước ấm theo tỉ lệ..." | ❌ No |
| General recommended dosage (no stated risk) | "Dùng 1–2 gói/ngày", "Lượng dùng đề nghị: 1–5 hộp/ngày" | ❌ No |
| Marketing / declared-content disclaimer | "Không bổ sung đường trong sản xuất", "Tìm hiểu thêm tại website..." | ❌ No |
| Manufacturer info | company address, website | ❌ No (may relate to `origin`, not `warning`) |

### 2.3 Two easily-confused distinctions

1. **Age-restriction warning vs. dosage-by-age**: "Dùng cho người trên 3
   tuổi." (a binary categorical restriction) **is** `warning`. But "Dưới 10
   tuổi: 2 ly/ngày." (a quantitative dosage table split by age bracket) is
   **not** — even though these two sentences often sit right next to each
   other on the same label, they're different categories. The draft
   frequently conflates them; split them apart on review.
2. **Date disclosure vs. expiry-related warning**: a sentence that *purely*
   states where/in what format the mfg/exp date is printed → not `warning`
   (belongs to `mfg_date`/`expiry_date`). A sentence that *commands or
   states a consequence* about using an expired product (e.g. "Không sử
   dụng sản phẩm đã hết hạn sử dụng.") → still a valid `warning`, since it
   passes the functional test (a)+(b). The draft sometimes duplicates the
   date-disclosure sentence into both `warning` and the date field — if you
   see that, remove it from `warning` and keep it only in the date field.

### 2.4 Exception

If a storage/usage sentence is paired with an explicit health consequence
(e.g. "Sau khi mở, không bảo quản lạnh có thể gây ngộ độc thực phẩm"), only
the health-consequence clause is extracted as `warning`; the pure
instruction part (temperature, duration) is not.

## 3. Other fields — notes for review

- **`product_name`/`nutrition`/`ingredient` being empty is usually not an
  error.** Many images only capture part of the packaging (the back panel,
  the ingredient panel, the nutrition panel) — no product name, nutrition
  table, or ingredient list in frame. Always look at the whole image before
  editing: only fill in information that's actually visible; never guess at
  content outside the frame, and never guess a full product name from a
  partially-visible logo/brand if the complete name isn't legible.
- **A relative `mfg_date` is valid.** Some labels state the manufacturing
  date relative to the expiry date instead of an absolute date (e.g. "NSX:
  06 tháng trước HSD", "Ngày sản xuất: 45 ngày trước hạn sử dụng"). Keep it
  verbatim as printed — don't compute an actual date.
- **Prefer Vietnamese on bilingual labels.** If a label prints both
  Vietnamese and English for the same content, take only the Vietnamese
  version (drop the duplicate English translation). Only take the English
  text if that field has no corresponding Vietnamese version.
- **Undocumented edge cases**: use best judgment and note it; if the same
  case recurs across multiple images, report it back so it can be added
  here for consistency across annotators.
