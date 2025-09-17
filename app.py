import streamlit as st
import io
import math
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageStat, ImageEnhance

# --- Konfiguracja strony ---
st.set_page_config(
    page_title="Analizator Oznaczeń Zamków",
    page_icon="🔎"
)

# --- Główny kod aplikacji (logika z poprzednich wersji) ---

# Parametry (można je dostosować w interfejsie lub zostawić stałe)
DPI = 150
MAX_WIDTH = 2000
ENHANCE = True
VISIBILITY_MEAN_THRESHOLD = 245
RED = (255, 0, 0)
STROKE = 2
FONT_CANDIDATES = ["DejaVuSans-Bold.ttf", "FreeSansBold.ttf", "Helvetica-Bold.ttf"]
DEFAULT_SHARPEN = 1.8
DEFAULT_CONTRAST = 1.2

# Funkcje pomocnicze (bez zmian)
def make_units(dpi: int):
    px_to_pt = 72.0 / dpi
    prox_thresh_pt = 10 * px_to_pt
    height_radius_pt = 60 * px_to_pt
    return px_to_pt, prox_thresh_pt, height_radius_pt

@st.cache_resource
def load_font(preferred_pt: int = 18):
    # Używamy cache, aby czcionka ładowała się tylko raz
    for name in FONT_CANDIDATES:
        try: return ImageFont.truetype(name, preferred_pt)
        except Exception: continue
    return ImageFont.load_default()

def pt_to_px(v, dpi): return v * (dpi / 72.0)
def bbox_pt_to_px(bbox, dpi): return tuple(pt_to_px(v, dpi) for v in bbox)

def fit_font_to_bbox(draw: ImageDraw.ImageDraw, label: str, bbox_px, base_pt=18):
    x0, y0, x1, y1 = bbox_px
    target_h = max(12, (y1 - y0) * 0.85)
    pt = base_pt
    font = load_font(pt)
    tb = draw.textbbox((0, 0), label, font=font)
    cur_h = max(1, tb[3] - tb[1])
    scale = target_h / cur_h
    pt = max(10, int(pt * scale))
    font = load_font(pt)
    return font

def tokens_are_pure_z(text: str) -> bool:
    tokens = text.split()
    if not tokens: return False
    for t in tokens:
        if not (t.startswith("Z") and t[1:].isdigit() and 1 <= len(t[1:]) <= 3): return False
    return True

def split_span_bbox(bbox, n: int):
    x0, y0, x1, y1 = bbox
    total_w = x1 - x0
    if total_w <= 0 or n <= 0: return [bbox] * max(1, n)
    gap = total_w * 0.05
    sub_w = (total_w - gap * (n - 1)) / n if n > 1 else total_w
    out = []
    cur_x = x0
    for _ in range(n):
        sx0, sx1 = cur_x, cur_x + sub_w
        out.append((sx0, y0, sx1, y1))
        cur_x = sx1 + gap
    return out

def is_duplicate(existing, new_lbl, new_bbox, prox_thresh_pt):
    for lbl, bbox in existing:
        if lbl == new_lbl and abs(bbox[0] - new_bbox[0]) < prox_thresh_pt and abs(bbox[1] - new_bbox[1]) < prox_thresh_pt:
            return True
    return False

def parse_heights(text_dict):
    res = []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text","").strip()
                if (t.startswith("H=") and t[2:].isdigit()) or (t.endswith("cm") and t[:-2].isdigit()):
                    res.append((t, span["bbox"]))
    return res

def nearest_height_for(bbox, heights, height_radius_pt):
    bx, by = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    best, best_dist = "", height_radius_pt
    for htxt, hbbox in heights:
        hx, hy = (hbbox[0] + hbbox[2]) / 2.0, (hbbox[1] + hbbox[3]) / 2.0
        dist = math.hypot(hx - bx, hy - by)
        if dist <= height_radius_pt and dist < best_dist:
            best_dist = dist
            if htxt.startswith("H="): best = htxt[2:]
            elif htxt.endswith("cm"): best = htxt[:-2]
    return best

def visible_on_image(img, bbox_pt, dpi):
    x0, y0, x1, y1 = bbox_pt_to_px(bbox_pt, dpi)
    if not (0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height): return False
    pad = 1
    crop_box = (max(0, int(x0) - pad), max(0, int(y0) - pad), min(img.width, int(math.ceil(x1)) + pad), min(img.height, int(math.ceil(y1)) + pad))
    crop = img.crop(crop_box).convert("L")
    mean_val = ImageStat.Stat(crop).mean[0]
    return mean_val < VISIBILITY_MEAN_THRESHOLD

def render_page_image(page, dpi):
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

def postprocess_png(img: Image.Image, max_width: int, enhance: bool) -> Image.Image:
    if max_width and img.width > max_width:
        scale = max_width / float(img.width)
        new_size = (max_width, int(round(img.height * scale)))
        img = img.resize(new_size, resample=Image.Resampling.LANCZOS)
    if enhance:
        img = ImageEnhance.Sharpness(img).enhance(DEFAULT_SHARPEN)
        img = ImageEnhance.Contrast(img).enhance(DEFAULT_CONTRAST)
    return img

# --- Główna funkcja przetwarzająca ---
def process_pdf_to_results(pdf_bytes):
    _, prox_thresh_pt, height_radius_pt = make_units(DPI)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_rows, page_images = [], []

    for page in doc:
        text_dict = page.get_text("dict")
        page_labels = []
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if tokens_are_pure_z(txt):
                        tokens = txt.split()
                        sub_boxes = split_span_bbox(span["bbox"], len(tokens)) if len(tokens) > 1 else [span["bbox"]]
                        page_labels.extend(zip(tokens, sub_boxes))
        dedup = []
        for lbl, bbox in page_labels:
            if not is_duplicate(dedup, lbl, bbox, prox_thresh_pt): dedup.append((lbl, bbox))
        img = render_page_image(page, DPI)
        heights = parse_heights(text_dict)
        accepted = []
        for lbl, bbox in dedup:
            if visible_on_image(img, bbox, DPI):
                htxt = nearest_height_for(bbox, heights, height_radius_pt)
                accepted.append((lbl, bbox, htxt))
        draw = ImageDraw.Draw(img)
        for lbl, bbox, _h in accepted:
            x0, y0, x1, y1 = bbox_pt_to_px(bbox, DPI)
            draw.rectangle([x0, y0, x1, y1], outline=RED, width=STROKE)
            font = fit_font_to_bbox(draw, lbl, (x0, y0, x1, y1), base_pt=18)
            tb = draw.textbbox((0, 0), lbl, font=font)
            draw.text((x0, y0 - (tb[3] - tb[1])), lbl, fill=RED, font=font)
        all_rows.extend([(lbl, htxt) for lbl, _bbox, htxt in accepted])
        page_images.append(img)
    
    if not page_images: return None, None

    total_h = sum(im.height for im in page_images)
    max_w = max(im.width for im in page_images)
    combined_img = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    for im in page_images:
        combined_img.paste(im, (0, y)); y += im.height
    final_img = postprocess_png(combined_img, max_width=MAX_WIDTH, enhance=ENHANCE)

    df = pd.DataFrame(all_rows, columns=["Oznaczenie", "Wysokość"])
    if not df.empty:
        df_summary = (df.groupby(["Oznaczenie", "Wysokość"]).size().reset_index(name="Liczba wystąpień"))
        df_summary["num"] = df_summary["Oznaczenie"].str.extract(r"Z(\d+)").astype(int)
        df_summary["hnum"] = pd.to_numeric(df_summary["Wysokość"], errors="coerce")
        df_summary = df_summary.sort_values(["num", "hnum"], na_position="last").drop(columns=["num", "hnum"])
    else:
        df_summary = pd.DataFrame(columns=["Oznaczenie", "Wysokość", "Liczba wystąpień"])
    
    return final_img, df_summary

# --- Interfejs użytkownika w Streamlit ---
st.title("🔎 Analizator Oznaczeń Zamków")
st.write("Wgraj plik PDF z rysunkiem technicznym, aby automatycznie znaleźć oznaczenia typu 'Z...' i wygenerować zestawienie.")

uploaded_file = st.file_uploader("Wybierz plik PDF", type="pdf")

if uploaded_file is not None:
    # Wczytanie pliku do pamięci
    pdf_bytes = uploaded_file.getvalue()
    
    with st.spinner('Trwa analiza pliku... To może potrwać chwilę.'):
        try:
            # Przetwarzanie pliku
            annotated_image, summary_df = process_pdf_to_results(pdf_bytes)

            if annotated_image and not summary_df.empty:
                st.success("Analiza zakończona pomyślnie!")
                
                # Wyświetlenie obrazu
                st.subheader("Obraz z oznaczeniami")
                st.image(annotated_image, caption="Wynik analizy", use_column_width=True)

                # Wyświetlenie tabeli z danymi
                st.subheader("Zestawienie wyników")
                st.dataframe(summary_df)

                # Przygotowanie plików do pobrania
                img_buffer = io.BytesIO()
                annotated_image.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()

                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    summary_df.to_excel(writer, index=False, sheet_name='Zestawienie')
                excel_bytes = excel_buffer.getvalue()
                
                # Przyciski do pobierania
                st.download_button(
                    label="Pobierz obraz (.png)",
                    data=img_bytes,
                    file_name=f"{Path(uploaded_file.name).stem}_oznaczenia.png",
                    mime="image/png"
                )
                st.download_button(
                    label="Pobierz zestawienie (.xlsx)",
                    data=excel_bytes,
                    file_name=f"{Path(uploaded_file.name).stem}_zestawienie.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Nie znaleziono żadnych oznaczeń w podanym pliku PDF.")

        except Exception as e:
            st.error(f"Wystąpił błąd podczas przetwarzania pliku: {e}")
            st.error("Upewnij się, że plik PDF nie jest uszkodzony i zawiera warstwę tekstową.")

else:
    st.info("Oczekuję na wgranie pliku PDF.")
