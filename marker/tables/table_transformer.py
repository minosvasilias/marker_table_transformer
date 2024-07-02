
from transformers import TableTransformerForObjectDetection
import torch
from torchvision import transforms
from PIL import ImageDraw, Image
import os
from tqdm.auto import tqdm
import time
import csv
from marker.schema.page import Page
from typing import List
from marker.tables.utils import sort_table_blocks
from marker.pdf.images import render_image
from marker.settings import settings
import logging

logger = logging.getLogger(__name__)

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image


# Load table transformer model
def load_model(model_name="microsoft/table-structure-recognition-v1.1-all"):
    model = TableTransformerForObjectDetection.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model


# Extract single table from page
def get_table_table_transformer(doc, page: Page, table_box, model, debug_mode=False) -> List[List[str]]:
    logger.debug(f"Processing page {page.pnum} with table box {table_box}")
    # Get page image in appropriate resolution
    page_image = render_image(doc[page.pnum], dpi=settings.SURYA_LAYOUT_DPI)

    # Scale image to page.bbox
    new_width = page.bbox[2] - page.bbox[0]
    new_height = page.bbox[3] - page.bbox[1]
    page_image = page_image.resize((int(new_width), int(new_height)))

    if debug_mode:
        store_page_image(page_image, table_box, page)

    # Get cropped table image and sort text blocks
    table_img = get_table_img(table_box, page_image)
    blocks = sort_table_blocks(page.blocks)

    # Get text lines and simplified text spans
    text_lines = [line for block in blocks for line in block.lines]
    text_spans = [{
        "text": span.text,
        "bbox": span.bbox
    } for line in text_lines for span in line.spans]

    # Find text lines that lie within table bbox
    text_lines = find_text_within_table(table_box, text_spans)

    # Ensure text bboxes are relative to table bbox and add padding
    table_img, text_lines = adjust_bboxes(table_box, text_lines, table_img, add_padding=True)
    
    # Parse table with table transformer
    table_data = parse_table(table_img, text_lines, model)

    # Convert table data to JSON list
    table_rows = []
    for row in table_data:
        table_rows.append(table_data[row])
    
    return table_to_json(table_rows, True)


# Convert table rows to JSON, amend headers if necessary
def table_to_json(table_rows, sort_by_rows=False):
    if len(table_rows) == 0:
        return {}
    headers = []
    for i, item in enumerate(table_rows[0]):
        if item.strip():
            headers.append(item)
        else:
            headers.append(f"NO_HEADER_{i}")
    if sort_by_rows:
        table_json = []
        for row in table_rows[1:]:
            row_obj = {}
            for i, item in enumerate(row):
                row_obj[headers[i]] = item
            table_json.append(row_obj)
    else:
        table_json = {}
        for i, header in enumerate(headers):
            table_json[header] = []
            for row in table_rows[1:]:
                table_json[header].append(row[i])
    return table_json


# Find text bboxes that lie safely within table bbox
def find_text_within_table(table_bbox, text_spans):
    SAFE_OVERLAP = 85
    lines = []
    for span in text_spans:
        overlap, _, _ = bbox_overlap(table_bbox, span["bbox"])
        if overlap > SAFE_OVERLAP:
            lines.append({
                "text": span["text"],
                "bbox": span["bbox"],
            })
    return lines

# Crop table image from page image
def get_table_img(table_bbox, page_img):
    table_img = page_img.crop(table_bbox)
    return table_img

# Subtract table bbox from text lines and add padding if necessary
def adjust_bboxes(table_bbox, text_lines, table_img, add_padding=False):
    # Adjust line bboxes to be relative to the table bbox
    for line in text_lines:
        line["bbox"][0] -= table_bbox[0]
        line["bbox"][1] -= table_bbox[1]
        line["bbox"][2] -= table_bbox[0]
        line["bbox"][3] -= table_bbox[1]

    if add_padding:
        # Add padding to the table image
        padding_horizontal = 40
        padding_vertical = 0
        canvas = Image.new("RGB", (table_img.width + 2 * padding_horizontal, table_img.height + 2 * padding_vertical), "white")
        x_offset = (canvas.width - table_img.width) // 2
        y_offset = (canvas.height - table_img.height) // 2
        canvas.paste(table_img, (x_offset, y_offset))
        table_img = canvas

        # Adjust line bboxes to be relative to the padded table bbox
        for line in text_lines:
            line["bbox"][0] += padding_horizontal
            line["bbox"][1] += padding_vertical
            line["bbox"][2] += padding_horizontal
            line["bbox"][3] += padding_vertical
        
    return table_img, text_lines


# Calculate overlap between two bounding boxes
# as well as left and right cut percentages
def bbox_overlap(bbox1, bbox2, debug_mode=False):
    x1, y1, x1e, y1e = bbox1
    x2, y2, x2e, y2e = bbox2

    # Calculate the (x, y) coordinates of the intersection rectangle
    left = max(x1, x2)
    right = min(x1e, x2e)
    top = max(y1, y2)
    bottom = min(y1e, y2e)

    # Calculate width and height of the intersection rectangle
    inter_width = max(0, right - left)
    inter_height = max(0, bottom - top)

    # Calculate the area of intersection rectangle
    overlap_area = inter_width * inter_height
    bbox2_area = (x2e - x2) * (y2e - y2)

    if overlap_area == 0 or bbox2_area == 0:
        return 0, 0, 0

    # Calculate the percentage of overlap, i.e. the percentage of bbox2_area that is part of overlap_area
    overlap = (overlap_area / bbox2_area) * 100

    if debug_mode:
        visualize_overlap(bbox1, bbox2, left, top, inter_width, inter_height, right, bottom, overlap)
    
    # Calculate percentage bbox is cut off to left and right
    left_cut = (left - x2) / (x2e - x2) * 100
    right_cut = (x2e - right) / (x2e - x2) * 100

    return overlap, left_cut, right_cut




# Parse table image using table transformer
def parse_table(table_img, text_lines, model, debug_mode=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # From: https://github.com/microsoft/table-transformer/blob/main/src/inference.py#L45
    structure_transform = transforms.Compose([
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pixel_values = structure_transform(table_img).unsqueeze(0)
    pixel_values = pixel_values.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(pixel_values)

    # Update id2label to include "no object"
    structure_id2label = model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    # Convert outputs to usable objects
    cells = outputs_to_objects(outputs, table_img.size, structure_id2label)

    if debug_mode:
        visualize_table(table_img, cells, text_lines)

    # Get cell coordinates by row
    cell_coordinates = get_cell_coordinates_by_row(cells)

    # Associate with OCR
    data = find_text(cell_coordinates, text_lines, debug_mode)

    # Clean data
    data = clean_data(data)

    if debug_mode:
        store_as_csv(data)

    return data


# Remove all empty rows and columns
def clean_data(data):
    def is_empty(value):
        return value.replace("\n", "").strip() == ""
    
    # Remove empty rows
    data = {row: row_data for row, row_data in data.items() if not all(is_empty(value) for value in row_data)}

    # Remove empty columns
    num_columns = min(len(row_data) for row_data in data.values())
    for col in range(num_columns - 1, -1, -1):
        if all(is_empty(row[col]) for row in data.values()):
            for row in data:
                del data[row][col]
    return data


# Find highest overlap text bboxes for each cell
def find_text(cell_coordinates, text_lines, debug_mode=False):
    data = {}
    max_num_columns = 0
    MIN_OVERLAP = 30
    
    # First pass: Calculate overlaps for all cells
    cell_overlaps = {}
    for idx, row in enumerate(tqdm(cell_coordinates)):
        cell_overlaps[idx] = []
        for cell in row["cells"]:
            cell_bbox = cell["cell"]
            cell_texts = []
            for text_idx, text_line in enumerate(text_lines):
                text_bbox = text_line["bbox"]
                overlap, left_cut, right_cut = bbox_overlap(cell_bbox, text_bbox, debug_mode)
                if overlap > MIN_OVERLAP:
                    # Store y-coordinate for vertical ordering
                    cell_texts.append((overlap, text_idx, text_line["text"], text_bbox[1]))
            # Sort primarily by y-coordinate (ascending), then by overlap (descending)
            cell_overlaps[idx].append(sorted(cell_texts, key=lambda x: (x[3], -x[0])))

    # Second pass: Assign text to cells
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell_idx, cell_texts in enumerate(cell_overlaps[idx]):
            if not cell_texts:
                row_text.append("")
                continue
            
            assigned_texts = []
            for overlap, text_idx, text, _ in cell_texts:
                if not text_lines[text_idx].get("used", False):
                    assigned_texts.append(text)
                    text_lines[text_idx]["used"] = True
            
            if assigned_texts:
                row_text.append("\n".join(assigned_texts))
        
        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)
        data[idx] = row_text

    # Pad rows to ensure all have the same number of columns
    for row, row_data in data.items():
        if len(row_data) < max_num_columns:
            data[row] = row_data + ["" for _ in range(max_num_columns - len(row_data))]

    return data


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


# Find and sort cell coordinates
def get_cell_coordinates_by_row(cells):
    # Extract rows and columns
    rows = [entry for entry in cells if entry['label'] == 'table row']
    columns = [entry for entry in cells if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates



# DEBUG: Visualize bbox overlaps
def visualize_overlap(bbox1, bbox2, left, top, inter_width, inter_height, right, bottom, overlap):
    try:
        if overlap > 50 and inter_width > 0 and inter_height > 0:
            img = Image.new('RGB', (1000, 1000), color='white')
            draw = ImageDraw.Draw(img)
            draw.rectangle(bbox1, outline="red")
            draw.rectangle(bbox2, outline="blue")
            draw.rectangle([left, top, right, bottom], outline="green")
            import time
            cur_time = str(int(time.time()))
            if not os.path.exists("temp/table_transformer"):
                os.makedirs("temp/table_transformer")
            img.save(f"temp/table_transformer/overlap_{cur_time}.png")
    except Exception as e:
        logger.info(f"Error visualizing overlap: {e}")



# DEBUG: Visualize table and cell bboxes
def visualize_table(table_img, cells, text_lines):
    cropped_table_visualized = table_img.copy()
    draw = ImageDraw.Draw(cropped_table_visualized)

    for cell in cells:
        draw.rectangle(cell["bbox"], outline="red")

    for line in text_lines:
        draw.rectangle(line["bbox"], outline="blue")

    # Store the visualized image
    temp_dir = "temp/table_transformer"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    cropped_table_visualized.save(f"{temp_dir}/visualized_{time.time()}.png")


# DEBUG: Store data as CSV
def store_as_csv(data, temp_dir = "temp/table_transformer"):
    try:
        with open(f'{temp_dir}/{time.time()}_ocr.csv', 'w') as result_file:
            wr = csv.writer(result_file, dialect='excel')

            for row, row_text in data.items():
                wr.writerow(row_text)
    except Exception as e:
        logger.info(f"Error storing as CSV: {e}")

# DEBUG: Store page image with table bbox
def store_page_image(page_image, table_box, page):
    img = page_image.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle(table_box, outline="red")
    if not os.path.exists("temp/table_transformer"):
        os.makedirs("temp/table_transformer")
    img.save(f"temp/table_transformer/page_{page.pnum}_{time.time()}_table_bbox.png")