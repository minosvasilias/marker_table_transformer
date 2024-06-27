
from transformers import TableTransformerForObjectDetection
import torch
from torchvision import transforms
from PIL import ImageDraw, Image
import os
from tqdm.auto import tqdm
import numpy as np
import time
import csv
from marker.schema.page import Page
from typing import List
from marker.tables.utils import sort_table_blocks, replace_dots, replace_newlines
from marker.pdf.images import render_image
from marker.settings import settings
from marker.schema.bbox import merge_boxes, box_intersection_pct, rescale_bbox

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image



def load_model(model_name="microsoft/table-structure-recognition-v1.1-all"):
    model = TableTransformerForObjectDetection.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model


def get_table_table_transformer(doc, page: Page, table_box, model) -> List[List[str]]:
    pnum = page.pnum
    print(f"Processing page {pnum} with table box {table_box}")
    page_image = render_image(doc[pnum], dpi=settings.SURYA_LAYOUT_DPI)
    # Scale image to page.bbox
    new_width = page.bbox[2] - page.bbox[0]
    new_height = page.bbox[3] - page.bbox[1]
    page_image = page_image.resize((int(new_width), int(new_height)))
    # Store image of table bbox
    img = page_image.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle(table_box, outline="red")
    if not os.path.exists("temp/table_transformer"):
        os.makedirs("temp/table_transformer")
    img.save(f"temp/table_transformer/page_{pnum}_{time.time()}_table_bbox.png")

    print(f"Page image size: {page_image.size}")
    table_img = get_table_img(table_box, page_image)
    print(f"Table image size: {table_img.size}")
    blocks = sort_table_blocks(page.blocks)
    #print(f"Blocks: {len(blocks)}")
    text_lines = [line for block in blocks for line in block.lines]
    text_spans = [{
        "text": span.text,
        "bbox": span.bbox
    } for line in text_lines for span in line.spans]
    #print(f"Text spans: {len(text_spans)}")
    text_lines = find_text_within_table(table_box, text_spans)
    print(f"Text lines within table: {len(text_lines)}")
    table_img, text_lines = adjust_bboxes(table_box, text_lines, table_img, add_padding=True)
    print(f"Adjusted table image size: {table_img.size}")
    table_name = f"temp/table_transformer/table_{pnum}.png"
    table_data = parse_table(table_name, table_img, text_lines, model)
    #print(f"Table data: {table_data}")
    table_rows = []
    for row in table_data:
        table_rows.append(table_data[row])
    print(f"Table rows: {table_rows}")
    return table_to_json(table_rows, True)


        
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



def find_text_within_table(table_bbox, text_spans):
    lines = []
    for span in text_spans:
        overlap, _, _ = bbox_overlap(table_bbox, span["bbox"])
        if overlap > 85:
            # If it has text att
            lines.append({
                "text": span["text"],
                "bbox": span["bbox"],
            })
    return lines

def get_table_img(table_bbox, page_img):
    table_img = page_img.crop(table_bbox)
    return table_img

def adjust_bboxes(table_bbox, text_lines, table_img, add_padding=False):
    # Adjust line bboxes to be relative to the table bbox
    print(f"Removing table bbox {table_bbox} from text lines...")
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

    #print(f"Overlap: {overlap} between {bbox1} and {bbox2} with overlap area: {[left, top, inter_width, inter_height]}")

    if False:
        visualize_overlap(bbox1, bbox2, left, top, inter_width, inter_height, right, bottom, overlap)
    
    # Calculate percentage bbox is cut off to left and right
    left_cut = (left - x2) / (x2e - x2) * 100
    right_cut = (x2e - right) / (x2e - x2) * 100

    return overlap, left_cut, right_cut





def parse_table(table_name, table_img, text_lines, model, debug_mode=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Processing {table_name} with text lines:\n{text_lines}")

    structure_transform = transforms.Compose([
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pixel_values = structure_transform(table_img).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    print(pixel_values.shape)

    # forward pass
    with torch.no_grad():
        outputs = model(pixel_values)

    # update id2label to include "no object"
    structure_id2label = model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, table_img.size, structure_id2label)
    print(cells)

    if debug_mode:
        visualize_table(table_name, table_img, cells, text_lines)

    # Get cell coordinates by row
    cell_coordinates = get_cell_coordinates_by_row(cells)

    # Associate with OCR
    data = find_text(cell_coordinates, text_lines, table_img, debug_mode)

    if debug_mode:
        store_as_csv(data, "temp/table_transformer", table_name.split("/")[-1])

    return data


def find_text(cell_coordinates, text_lines, table_img, debug_mode=False):
    data = {}
    max_num_columns = 0

    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell in row["cells"]:
            cell_image = np.array(table_img.crop(cell["cell"]))
            cell_bbox = cell["cell"]
            overlapping_texts = []
            MIN_OVERLAP = 60
            SAFE_OVERLAP = 85
            largest_overlap = MIN_OVERLAP
            largest_overlap_text_line = None
            for text_line in text_lines:
                if "used" in text_line and text_line["used"]:
                    continue
                text_bbox = text_line["bbox"]
                overlap, left_cut, right_cut = bbox_overlap(cell_bbox, text_bbox, debug_mode)
                #print(f"Overlap: {overlap} with left cut: {left_cut} and right cut: {right_cut} for text: {text_line['text']}")
                #print(f"Overlap: {overlap} between {cell_bbox} and {text_bbox}: {text_line['text']}")
                if overlap > largest_overlap:
                    largest_overlap = overlap
                    largest_overlap_text_line = text_line
                if overlap > SAFE_OVERLAP:
                    overlapping_texts.append(text_line)
                    text_line["used"] = True
            if largest_overlap_text_line:
                largest_overlap_text = largest_overlap_text_line["text"]
                largest_overlap_text_line["used"] = True
                if len(overlapping_texts) > 1:
                    largest_overlap_text = "\n".join([text_line["text"] for text_line in overlapping_texts])
            else:
                largest_overlap_text = ""
            row_text.append(largest_overlap_text)
        # print(f"Row text: {row_text}")
        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)
        data[idx] = row_text

    # print("Max number of columns:", max_num_columns)

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

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




def visualize_overlap(bbox1, bbox2, left, top, inter_width, inter_height, right, bottom, overlap):
    # Visualize
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
        print(f"Error visualizing overlap: {e}")



def visualize_table(table_name, table_img, cells, text_lines):
    # Visualize
    cropped_table_visualized = table_img.copy()
    draw = ImageDraw.Draw(cropped_table_visualized)

    for cell in cells:
        draw.rectangle(cell["bbox"], outline="red")

    for line in text_lines:
        draw.rectangle(line["bbox"], outline="blue")

    # Store the visualized image
    original_name = table_name.split("/")[-1]
    original_name = original_name.split(".")[0]
    temp_dir = "temp/table_transformer"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    cropped_table_visualized.save(f"{temp_dir}/visualized_{original_name}_{time.time()}.png")


def store_as_csv(data, temp_dir, original_name):
    try:
        # Store as CSV
        with open(f'{temp_dir}/{original_name}_{time.time()}_ocr.csv', 'w') as result_file:
            wr = csv.writer(result_file, dialect='excel')

            for row, row_text in data.items():
                wr.writerow(row_text)
    except Exception as e:
        print(f"Error storing as CSV: {e}")
