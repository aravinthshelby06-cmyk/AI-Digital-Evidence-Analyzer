import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import hashlib
from datetime import datetime
from io import BytesIO
from collections import Counter
import tempfile
import os
import cv2

from ultralytics import YOLOE

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Digital Evidence Analyzer",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #777;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🔎 AI Digital Evidence Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an evidence image or video to detect visible objects using AI.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD YOLOE MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = YOLOE("yoloe-26s-seg-pf.pt")

    return model


# =========================================================
# SHA-256
# =========================================================

def calculate_sha256(file_bytes):

    return hashlib.sha256(file_bytes).hexdigest()


# =========================================================
# IMAGE VALIDATION
# =========================================================

def validate_image(image):

    width, height = image.size

    if width < 100 or height < 100:
        return False, "Image is too small."

    if width > 12000 or height > 12000:
        return False, "Image resolution is too large."

    return True, ""


# =========================================================
# CATEGORY MAPPING
# =========================================================

def get_category(name):

    name_lower = name.lower()

    if any(x in name_lower for x in [
        "person", "man", "woman",
        "boy", "girl", "human"
    ]):
        return "Person"

    if any(x in name_lower for x in [
        "gun", "pistol", "rifle",
        "firearm", "weapon"
    ]):
        return "Weapon"

    if any(x in name_lower for x in [
        "knife", "dagger", "blade"
    ]):
        return "Knife"

    if any(x in name_lower for x in [
        "car", "vehicle", "automobile",
        "truck", "bus", "motorcycle",
        "bike"
    ]):
        return "Vehicle"

    if any(x in name_lower for x in [
        "bag", "backpack",
        "handbag", "suitcase"
    ]):
        return "Bag"

    if any(x in name_lower for x in [
        "phone", "mobile",
        "cell phone", "smartphone"
    ]):
        return "Mobile / Phone"

    if any(x in name_lower for x in [
        "laptop", "computer",
        "keyboard", "monitor",
        "tablet"
    ]):
        return "Electronic Device"

    if any(x in name_lower for x in [
        "document", "paper", "book"
    ]):
        return "Document"

    if any(x in name_lower for x in [
        "bottle", "cup",
        "glass", "container"
    ]):
        return "Container"

    if any(x in name_lower for x in [
        "shirt", "dress", "pants",
        "clothing", "shoe",
        "hat", "cap", "glove"
    ]):
        return "Clothing"

    if any(x in name_lower for x in [
        "tool", "hammer",
        "screwdriver", "wrench",
        "handsaw", "saw"
    ]):
        return "Tool"

    return "Other"


# =========================================================
# UNWANTED LABEL FILTER
# =========================================================

SUSPICIOUS_LABELS = {

    "login",
    "logout",
    "sign in",
    "sign up",
    "website",
    "web page",
    "webpage",
    "software",
    "application",
    "app",
    "user interface",
    "interface",
    "menu",
    "button",
    "icon",
    "notification",
    "username",
    "password",
    "account",
    "email address",
    "email",
    "url",
    "link",
    "3d rendering",
    "3d render",
    "3d cg rendering",
    "computer generated image",
    "digital rendering",
    "screen capture",
    "screen shot",
    "screenshot",
    "error message",
    "loading screen",
    "plot",
    "residential neighborhood",
    "neighborhood",
    "residential area",
    "urban area",
    "rural area",
    "landscape",
    "scenery",
    "background",
    "foreground"
}


def is_suspicious_label(name):

    label = name.lower().strip()

    if label in SUSPICIOUS_LABELS:
        return True

    for suspicious in SUSPICIOUS_LABELS:

        if suspicious in label:
            return True

    return False


# =========================================================
# IOU
# =========================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = max(
        0,
        box1[2] - box1[0]
    ) * max(
        0,
        box1[3] - box1[1]
    )

    area2 = max(
        0,
        box2[2] - box2[0]
    ) * max(
        0,
        box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


# =========================================================
# DUPLICATE FILTER
# =========================================================

def remove_duplicate_detections(detections):

    final_detections = []

    for detection in detections:

        duplicate = False

        for existing in final_detections:

            if (
                detection["name"].lower()
                ==
                existing["name"].lower()
            ):

                iou = calculate_iou(
                    detection["bbox"],
                    existing["bbox"]
                )

                if iou >= 0.80:

                    duplicate = True
                    break

        if not duplicate:

            final_detections.append(
                detection
            )

    return final_detections


# =========================================================
# DRAW IMAGE BOXES
# =========================================================

def draw_filtered_boxes(image, detections):

    output = image.copy()

    draw = ImageDraw.Draw(output)

    try:

        font = ImageFont.truetype(
            "arial.ttf",
            18
        )

    except:

        font = ImageFont.load_default()

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        label = (
            f"{detection['name']} "
            f"{detection['confidence']:.0%}"
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4
        )

        text_box = draw.textbbox(
            (x1, y1),
            label,
            font=font
        )

        draw.rectangle(
            text_box,
            fill="red"
        )

        draw.text(
            (x1, y1),
            label,
            fill="white",
            font=font
        )

    return output


# =========================================================
# IMAGE DETECTION EXTRACTION
# =========================================================

def extract_image_detections(
    result,
    image,
    confidence_threshold,
    show_low_confidence
):

    detections = []

    if result.boxes is None:
        return detections

    boxes = result.boxes

    image_area = (
        image.size[0] *
        image.size[1]
    )

    for i in range(len(boxes)):

        confidence = float(
            boxes.conf[i].item()
        )

        class_id = int(
            boxes.cls[i].item()
        )

        name = result.names.get(
            class_id,
            str(class_id)
        )

        # Remove unwanted digital/scene labels
        if is_suspicious_label(name):
            continue

        xyxy = boxes.xyxy[i].tolist()

        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])

        width = max(
            0,
            x2 - x1
        )

        height = max(
            0,
            y2 - y1
        )

        box_area = width * height

        area_ratio = (
            box_area / image_area
            if image_area > 0
            else 0
        )

        # Very tiny detections
        if area_ratio < 0.0005:
            continue

        if (
            not show_low_confidence
            and confidence < 0.35
        ):
            continue

        category = get_category(name)

        detections.append(
            {
                "name": name,
                "category": category,
                "confidence": confidence,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            }
        )

    return remove_duplicate_detections(
        detections
    )


# =========================================================
# PDF REPORT - IMAGE
# =========================================================

def create_image_pdf_report(
    file_name,
    timestamp,
    image_size,
    sha256,
    confidence,
    detections
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "AI Digital Evidence Analysis Report",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            f"<b>Evidence File:</b> {file_name}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Evidence Type:</b> Image",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Analysis Time:</b> {timestamp}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Image Resolution:</b> "
            f"{image_size[0]} x {image_size[1]}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Confidence Threshold:</b> "
            f"{confidence:.2f}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            f"<b>SHA-256 Evidence Hash:</b><br/>"
            f"{sha256}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "<b>Detected Objects</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    if detections:

        data = [
            [
                "No.",
                "Object",
                "Category",
                "Confidence",
                "Bounding Box"
            ]
        ]

        for index, detection in enumerate(
            detections,
            start=1
        ):

            bbox = detection["bbox"]

            bbox_text = (
                f"({bbox[0]}, {bbox[1]}) - "
                f"({bbox[2]}, {bbox[3]})"
            )

            data.append(
                [
                    str(index),
                    detection["name"],
                    detection["category"],
                    f"{detection['confidence']:.2f}",
                    bbox_text
                ]
            )

        table = Table(
            data,
            colWidths=[
                30,
                90,
                90,
                75,
                155
            ]
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.grey
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP"
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8
                    )
                ]
            )
        )

        story.append(table)

    else:

        story.append(
            Paragraph(
                "No objects detected above the "
                "selected confidence threshold.",
                styles["Normal"]
            )
        )

    story.append(
        Spacer(1, 25)
    )

    story.append(
        Paragraph(
            "<b>Important:</b> This report represents "
            "AI-assisted visual object detection only. "
            "It does not establish identity, ownership, "
            "intent, authenticity, cause, or any other "
            "forensic or legal conclusion.",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "Research / Demonstration Use",
            styles["Normal"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer


# =========================================================
# PDF REPORT - VIDEO
# =========================================================

def create_video_pdf_report(
    file_name,
    timestamp,
    duration,
    fps,
    total_frames,
    resolution,
    sha256,
    confidence,
    detections
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "AI Digital Evidence Video Analysis Report",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            f"<b>Evidence File:</b> {file_name}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            "<b>Evidence Type:</b> Video",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Analysis Time:</b> {timestamp}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Video Duration:</b> {duration:.2f} seconds",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>FPS:</b> {fps:.2f}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Total Frames:</b> {total_frames}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Resolution:</b> "
            f"{resolution[0]} x {resolution[1]}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Confidence Threshold:</b> "
            f"{confidence:.2f}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            f"<b>SHA-256 Evidence Hash:</b><br/>"
            f"{sha256}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "<b>Video Detection Events</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    if detections:

        data = [
            [
                "No.",
                "Object",
                "Confidence",
                "Timestamp",
                "Frame"
            ]
        ]

        # Limit report size
        report_detections = detections[:500]

        for index, detection in enumerate(
            report_detections,
            start=1
        ):

            data.append(
                [
                    str(index),
                    detection["name"],
                    f"{detection['confidence']:.2f}",
                    f"{detection['timestamp']:.2f}s",
                    str(detection["frame"])
                ]
            )

        table = Table(
            data,
            colWidths=[
                35,
                130,
                80,
                90,
                70
            ],
            repeatRows=1
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.grey
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP"
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8
                    )
                ]
            )
        )

        story.append(table)

        if len(detections) > 500:

            story.append(
                Spacer(1, 10)
            )

            story.append(
                Paragraph(
                    f"Only the first 500 detection events "
                    f"are included in the PDF. Total events: "
                    f"{len(detections)}.",
                    styles["Normal"]
                )
            )

    else:

        story.append(
            Paragraph(
                "No objects detected in the video.",
                styles["Normal"]
            )
        )

    story.append(
        Spacer(1, 25)
    )

    story.append(
        Paragraph(
            "<b>Important:</b> This report represents "
            "AI-assisted visual object detection only. "
            "It does not establish identity, ownership, "
            "intent, authenticity, cause, or any other "
            "forensic or legal conclusion.",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "Research / Demonstration Use",
            styles["Normal"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Analysis Settings")

confidence_threshold = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.15,
    max_value=0.80,
    value=0.35,
    step=0.05
)

image_size = st.sidebar.selectbox(
    "Analysis Image Size",
    [640, 768, 896, 1024, 1280],
    index=2
)

show_low_confidence = st.sidebar.checkbox(
    "Show lower-confidence detections",
    value=False
)

st.sidebar.markdown("---")

st.sidebar.info(
    "YOLOE prompt-free AI model is used for "
    "open-vocabulary object detection."
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📁 Upload Evidence",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


# =========================================================
# NO FILE
# =========================================================

if uploaded_file is None:

    st.info(
        "Upload an image or video evidence file to begin analysis."
    )

    st.stop()


# =========================================================
# FILE BYTES + HASH
# =========================================================

file_bytes = uploaded_file.getvalue()

sha256_hash = calculate_sha256(
    file_bytes
)

file_name = uploaded_file.name

file_extension = os.path.splitext(
    file_name
)[1].lower()


# =========================================================
# IMAGE ANALYSIS
# =========================================================

if file_extension in [
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
]:

    try:

        image = Image.open(
            BytesIO(file_bytes)
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Unable to open image: {e}"
        )

        st.stop()


    valid, error_message = validate_image(
        image
    )

    if not valid:

        st.error(error_message)

        st.stop()


    # =====================================================
    # INFORMATION
    # =====================================================

    st.markdown(
        "### 📄 Evidence Information"
    )

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.write(
            f"**File:** {file_name}"
        )

    with info_col2:

        st.write(
            f"**Resolution:** "
            f"{image.size[0]} × {image.size[1]}"
        )

    with info_col3:

        st.write(
            f"**Format:** {image.format}"
        )

    st.markdown(
        f"**🔐 SHA-256:** `{sha256_hash}`"
    )

    st.markdown("---")


    # =====================================================
    # ORIGINAL IMAGE
    # =====================================================

    st.subheader(
        "🖼️ Original Evidence"
    )

    st.image(
        image,
        use_container_width=True
    )


    # =====================================================
    # ANALYZE
    # =====================================================

    analyze = st.button(
        "🔎 Analyze Evidence",
        type="primary",
        use_container_width=True
    )


    if analyze:

        try:

            with st.spinner(
                "Loading YOLOE AI model..."
            ):

                model = load_model()

        except Exception as e:

            st.error(
                "YOLOE model could not be loaded."
            )

            st.exception(e)

            st.stop()


        try:

            with st.spinner(
                "AI is analyzing the evidence image..."
            ):

                results = model.predict(
                    source=image,
                    conf=confidence_threshold,
                    imgsz=image_size,
                    iou=0.45,
                    max_det=100,
                    verbose=False
                )

        except Exception as e:

            st.error(
                "Error during AI analysis."
            )

            st.exception(e)

            st.stop()


        result = results[0]


        # =================================================
        # DETECTIONS
        # =================================================

        detections = extract_image_detections(
            result,
            image,
            confidence_threshold,
            show_low_confidence
        )


        # =================================================
        # RESULT IMAGE
        # =================================================

        annotated_image = draw_filtered_boxes(
            image,
            detections
        )


        st.markdown("---")

        st.subheader(
            "🔍 AI Analysis Result"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.image(
                image,
                caption="Original Evidence",
                use_container_width=True
            )

        with result_col2:

            st.image(
                annotated_image,
                caption="Detected Evidence Objects",
                use_container_width=True
            )


        # =================================================
        # SUMMARY
        # =================================================

        st.markdown("---")

        st.subheader(
            "📊 Detection Summary"
        )

        counter = Counter(
            detection["name"]
            for detection in detections
        )

        category_counter = Counter(
            detection["category"]
            for detection in detections
        )

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.metric(
                "Objects Detected",
                len(detections)
            )

        with summary_col2:

            st.metric(
                "Unique Objects",
                len(counter)
            )

        with summary_col3:

            st.metric(
                "Categories",
                len(category_counter)
            )


        if counter:

            st.write(
                "### Detected Objects"
            )

            for name, count in counter.items():

                st.write(
                    f"• **{name}** : {count}"
                )

        else:

            st.warning(
                "No objects detected above the selected threshold."
            )


        # =================================================
        # CATEGORY SUMMARY
        # =================================================

        if category_counter:

            st.write(
                "### 🗂️ Object Categories"
            )

            for category, count in category_counter.items():

                st.write(
                    f"• **{category}** : {count}"
                )


        # =================================================
        # DETAILS
        # =================================================

        if detections:

            st.markdown("---")

            st.subheader(
                "🔍 Detection Details"
            )

            for index, detection in enumerate(
                detections,
                start=1
            ):

                bbox = detection["bbox"]

                st.write(
                    f"**{index}. {detection['name']}**  \n"
                    f"Category: `{detection['category']}`  \n"
                    f"Confidence: `{detection['confidence']:.2f}`  \n"
                    f"Bounding Box: `{bbox}`"
                )


        # =================================================
        # HASH
        # =================================================

        st.markdown("---")

        st.subheader(
            "🔐 Evidence Integrity"
        )

        st.write(
            "SHA-256 hash:"
        )

        st.code(
            sha256_hash
        )


        # =================================================
        # PDF
        # =================================================

        analysis_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        pdf = create_image_pdf_report(
            file_name=file_name,
            timestamp=analysis_time,
            image_size=image.size,
            sha256=sha256_hash,
            confidence=confidence_threshold,
            detections=detections
        )


        st.markdown("---")

        st.subheader(
            "📄 Evidence Report"
        )

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf,
            file_name="AI_Digital_Evidence_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# =========================================================
# VIDEO ANALYSIS
# =========================================================

elif file_extension in [
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
]:

    st.subheader(
        "🎥 Evidence Video"
    )

    st.video(
        file_bytes
    )

    st.markdown(
        f"**🔐 SHA-256:** `{sha256_hash}`"
    )

    st.markdown("---")


    analyze_video = st.button(
        "🎥 Analyze Video Evidence",
        type="primary",
        use_container_width=True
    )


    if analyze_video:

        # =================================================
        # SAVE VIDEO
        # =================================================

        input_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        )

        input_temp.write(
            file_bytes
        )

        input_temp.close()

        input_path = input_temp.name


        # =================================================
        # OPEN VIDEO
        # =================================================

        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():

            st.error(
                "Unable to open the uploaded video."
            )

            try:
                os.unlink(input_path)
            except:
                pass

            st.stop()


        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25.0


        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        duration = (
            total_frames / fps
            if fps > 0
            else 0
        )


        # =================================================
        # VIDEO INFORMATION
        # =================================================

        st.info(
            f"Duration: {duration:.2f} sec | "
            f"FPS: {fps:.2f} | "
            f"Frames: {total_frames} | "
            f"Resolution: {width} × {height}"
        )


        # =================================================
        # LOAD MODEL
        # =================================================

        try:

            with st.spinner(
                "Loading YOLOE AI model..."
            ):

                model = load_model()

        except Exception as e:

            cap.release()

            st.error(
                "YOLOE model could not be loaded."
            )

            st.exception(e)

            st.stop()


        # =================================================
        # OUTPUT VIDEO
        # =================================================

        output_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_temp.name

        output_temp.close()


        # =================================================
        # VIDEO WRITER
        # =================================================

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )


        # =================================================
        # UI
        # =================================================

        progress_bar = st.progress(
            0
        )

        status_text = st.empty()

        frame_preview = st.empty()


        # =================================================
        # DETECTION STORAGE
        # =================================================

        all_video_detections = []

        frame_number = 0


        # =================================================
        # FRAME PROCESSING
        # =================================================

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_number += 1


                # -----------------------------------------
                # Analyze current frame
                # -----------------------------------------

                results = model.predict(
                    source=frame,
                    conf=confidence_threshold,
                    imgsz=image_size,
                    iou=0.45,
                    max_det=50,
                    verbose=False
                )

                result = results[0]


                # -----------------------------------------
                # Frame detections
                # -----------------------------------------

                frame_detections = []

                if result.boxes is not None:

                    boxes = result.boxes

                    for i in range(len(boxes)):

                        confidence = float(
                            boxes.conf[i].item()
                        )

                        class_id = int(
                            boxes.cls[i].item()
                        )

                        name = result.names.get(
                            class_id,
                            str(class_id)
                        )


                        # Remove unwanted labels
                        if is_suspicious_label(name):
                            continue


                        if (
                            not show_low_confidence
                            and confidence < 0.35
                        ):
                            continue


                        xyxy = boxes.xyxy[i].tolist()

                        x1 = int(xyxy[0])
                        y1 = int(xyxy[1])
                        x2 = int(xyxy[2])
                        y2 = int(xyxy[3])


                        # Small object filter

                        box_width = max(
                            0,
                            x2 - x1
                        )

                        box_height = max(
                            0,
                            y2 - y1
                        )

                        box_area = (
                            box_width *
                            box_height
                        )

                        frame_area = (
                            width *
                            height
                        )

                        area_ratio = (
                            box_area / frame_area
                            if frame_area > 0
                            else 0
                        )

                        if area_ratio < 0.0005:
                            continue


                        category = get_category(
                            name
                        )


                        detection = {
                            "name": name,
                            "category": category,
                            "confidence": confidence,
                            "bbox": [
                                x1,
                                y1,
                                x2,
                                y2
                            ],
                            "frame": frame_number,
                            "timestamp": (
                                frame_number / fps
                            )
                        }


                        frame_detections.append(
                            detection
                        )


                    # Remove duplicate detections
                    # inside same frame

                    frame_detections = (
                        remove_duplicate_detections(
                            frame_detections
                        )
                    )


                # -----------------------------------------
                # Store detections
                # -----------------------------------------

                all_video_detections.extend(
                    frame_detections
                )


                # -----------------------------------------
                # Annotated frame
                # -----------------------------------------

                annotated_frame = frame.copy()


                for detection in frame_detections:

                    x1, y1, x2, y2 = (
                        detection["bbox"]
                    )

                    label = (
                        f"{detection['name']} "
                        f"{detection['confidence']:.0%}"
                    )


                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2
                    )


                    cv2.rectangle(
                        annotated_frame,
                        (
                            x1,
                            max(
                                0,
                                y1 - 25
                            )
                        ),
                        (
                            x1 + max(
                                100,
                                len(label) * 10
                            ),
                            y1
                        ),
                        (0, 0, 255),
                        -1
                    )


                    cv2.putText(
                        annotated_frame,
                        label,
                        (
                            x1 + 3,
                            max(
                                18,
                                y1 - 6
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )


                # -----------------------------------------
                # Write frame
                # -----------------------------------------

                writer.write(
                    annotated_frame
                )


                # -----------------------------------------
                # Preview every 5 frames
                # -----------------------------------------

                if frame_number % 5 == 0:

                    preview = cv2.cvtColor(
                        annotated_frame,
                        cv2.COLOR_BGR2RGB
                    )

                    frame_preview.image(
                        preview,
                        caption=(
                            f"Frame {frame_number}/"
                            f"{total_frames} | "
                            f"{frame_number / fps:.2f}s"
                        ),
                        channels="RGB"
                    )


                # -----------------------------------------
                # Progress
                # -----------------------------------------

                if total_frames > 0:

                    progress = (
                        frame_number /
                        total_frames
                    )

                else:

                    progress = 0


                progress_bar.progress(
                    min(
                        1.0,
                        progress
                    )
                )


                status_text.text(
                    f"🔍 Analyzing video... "
                    f"{frame_number}/{total_frames} "
                    f"frames"
                )


        except Exception as e:

            st.error(
                "Error during video analysis."
            )

            st.exception(e)

        finally:

            cap.release()
            writer.release()


        # =================================================
        # COMPLETE
        # =================================================

        progress_bar.progress(
            1.0
        )

        status_text.success(
            "✅ Video analysis completed."
        )


        # =================================================
        # VIDEO SUMMARY
        # =================================================

        st.markdown("---")

        st.subheader(
            "📊 Video Detection Summary"
        )


        object_counter = Counter(
            d["name"]
            for d in all_video_detections
        )

        category_counter = Counter(
            d["category"]
            for d in all_video_detections
        )


        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.metric(
                "Detection Events",
                len(all_video_detections)
            )

        with summary_col2:

            st.metric(
                "Object Classes",
                len(object_counter)
            )

        with summary_col3:

            st.metric(
                "Categories",
                len(category_counter)
            )


        # =================================================
        # OBJECT SUMMARY
        # =================================================

        if object_counter:

            st.write(
                "### Detected Objects"
            )

            for name, count in object_counter.items():

                st.write(
                    f"• **{name}** : "
                    f"{count} frame detections"
                )

        else:

            st.warning(
                "No objects detected in the video "
                "above the selected threshold."
            )


        # =================================================
        # CATEGORY SUMMARY
        # =================================================

        if category_counter:

            st.write(
                "### 🗂️ Categories"
            )

            for category, count in category_counter.items():

                st.write(
                    f"• **{category}** : {count}"
                )


        # =================================================
        # TIMESTAMP RESULTS
        # =================================================

        if all_video_detections:

            st.markdown("---")

            st.subheader(
                "⏱️ Detection Timestamps"
            )

            # Show first 100 events in UI

            visible_events = (
                all_video_detections[:100]
            )

            for index, detection in enumerate(
                visible_events,
                start=1
            ):

                st.write(
                    f"**{index}. "
                    f"{detection['name']}** — "
                    f"{detection['timestamp']:.2f}s — "
                    f"Frame {detection['frame']} — "
                    f"Confidence "
                    f"{detection['confidence']:.2f}"
                )


            if len(all_video_detections) > 100:

                st.info(
                    f"Showing first 100 events "
                    f"out of "
                    f"{len(all_video_detections)}."
                )


        # =================================================
        # VIDEO HASH
        # =================================================

        st.markdown("---")

        st.subheader(
            "🔐 Video Evidence Integrity"
        )

        st.write(
            "SHA-256 hash:"
        )

        st.code(
            sha256_hash
        )


        # =================================================
        # ANNOTATED VIDEO
        # =================================================

        st.markdown("---")

        st.subheader(
            "🎞️ AI Annotated Video"
        )


        if os.path.exists(output_path):

            try:

                with open(
                    output_path,
                    "rb"
                ) as video_file:

                    annotated_video_bytes = (
                        video_file.read()
                    )


                st.video(
                    annotated_video_bytes
                )


                st.download_button(
                    label="⬇️ Download Annotated Video",
                    data=annotated_video_bytes,
                    file_name=(
                        "AI_Detected_Evidence_Video.mp4"
                    ),
                    mime="video/mp4",
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Unable to display annotated video: {e}"
                )


        # =================================================
        # VIDEO PDF
        # =================================================

        analysis_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        video_pdf = create_video_pdf_report(
            file_name=file_name,
            timestamp=analysis_time,
            duration=duration,
            fps=fps,
            total_frames=total_frames,
            resolution=(width, height),
            sha256=sha256_hash,
            confidence=confidence_threshold,
            detections=all_video_detections
        )


        st.markdown("---")

        st.subheader(
            "📄 Video Evidence Report"
        )


        st.download_button(
            label="⬇️ Download Video PDF Report",
            data=video_pdf,
            file_name=(
                "AI_Digital_Video_Evidence_Report.pdf"
            ),
            mime="application/pdf",
            use_container_width=True
        )


        # =================================================
        # CLEANUP
        # =================================================

        try:

            os.unlink(
                input_path
            )

        except:

            pass


        try:

            os.unlink(
                output_path
            )

        except:

            pass


# =========================================================
# INVALID FILE
# =========================================================

else:

    st.error(
        "Unsupported file format. "
        "Please upload JPG, JPEG, PNG, WEBP, MP4, AVI, MOV or MKV."
    )